from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from api.auth import verify_password, fake_users_db, create_access_token, get_current_user, timedelta, ACCESS_TOKEN_EXPIRE_MINUTES
from api.schemas import Token, PredictPayload
from api.deps import get_ml_cache, limiter
import pandas as pd
import numpy as np
import torch

router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/risk-score/{user_id}")
@limiter.limit("20/minute")
async def get_risk_score(request: Request, user_id: int, current_user: dict = Depends(get_current_user), cache=Depends(get_ml_cache)):
    if user_id not in cache.tx_agg.index:
        raise HTTPException(status_code=404, detail="User not found")
        
    features = cache.tx_agg.loc[user_id][cache.feature_names].values.reshape(1, -1)
    scaled_features = cache.scaler.transform(features)
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
    
    with torch.no_grad():
        preds = cache.model(tensor_features)
        predicted_class = torch.argmax(preds, dim=1).item()
        
    state_names = {0: "Casual", 1: "Frequent", 2: "Compulsive", 3: "Crisis"}
    return {
        "user_id": user_id,
        "risk_tier_id": predicted_class,
        "risk_tier_name": state_names[predicted_class]
    }

@router.get("/state/{user_id}")
@limiter.limit("20/minute")
async def get_state(request: Request, user_id: int, current_user: dict = Depends(get_current_user), cache=Depends(get_ml_cache)):
    if user_id not in cache.labels.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    proxy_score = int(cache.labels.loc[user_id, 'total_score'])
    
    if proxy_score <= 2: state = "Casual"
    elif proxy_score <= 5: state = "Frequent"
    elif proxy_score <= 7: state = "Compulsive"
    else: state = "Crisis"
    
    return {
        "user_id": user_id,
        "current_behavioral_state": state,
        "dsm5_proxy_score": proxy_score,
        "transition_history": ["Data available in offline clustering analysis"]
    }

@router.get("/explain/{user_id}")
@limiter.limit("10/minute")
async def get_explanation(request: Request, user_id: int, current_user: dict = Depends(get_current_user), cache=Depends(get_ml_cache)):
    if user_id not in cache.tx_agg.index:
        raise HTTPException(status_code=404, detail="User not found")
        
    features = cache.tx_agg.loc[user_id][cache.feature_names].values.reshape(1, -1)
    scaled_features = cache.scaler.transform(features)
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
    
    shap_values = cache.explainer.shap_values(tensor_features)
    
    with torch.no_grad():
        predicted_class = torch.argmax(cache.model(tensor_features), dim=1).item()
        
    if isinstance(shap_values, list):
        sv = shap_values[predicted_class][0]
    else:
        sv = shap_values[0, :, predicted_class]
        
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    
    explanations = []
    for idx in top_idx:
        feat = cache.feature_names[idx]
        impact = sv[idx]
        direction = "increasing" if impact > 0 else "decreasing"
        explanations.append(f"The feature '{feat}' is significantly {direction} the user's risk profile (impact: {impact:+.4f}).")
        
    return {
        "user_id": user_id,
        "predicted_state": predicted_class,
        "plain_english_explanation": explanations
    }

@router.get("/platform-report")
@limiter.limit("5/minute")
async def platform_report(request: Request, current_user: dict = Depends(get_current_user)):
    from models.dark_pattern_scorer import generate_report
    df = generate_report()
    return {
        "status": "success",
        "data": df.to_dict(orient="records")
    }

@router.post("/alert/{user_id}")
@limiter.limit("5/minute")
async def trigger_alert(request: Request, user_id: int, current_user: dict = Depends(get_current_user), cache=Depends(get_ml_cache)):
    if user_id not in cache.labels.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    score = cache.labels.loc[user_id, 'total_score']
    if score >= 6:
        return {"user_id": user_id, "alert_triggered": True, "intervention_level": "High/Crisis"}
    return {"user_id": user_id, "alert_triggered": False, "message": "User not in compulsive/crisis state."}

@router.post("/score/predict")
@limiter.limit("10/minute")
async def real_time_predict(request: Request, payload: PredictPayload, current_user: dict = Depends(get_current_user), cache=Depends(get_ml_cache)):
    df = pd.DataFrame([t.dict() for t in payload.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="No transactions provided")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    txn_count = len(df)

    time_span = (df['timestamp'].max() - df['timestamp'].min()).days
    days_since_first = float(time_span if time_span > 0 else 1.0)

    spending_velocity = float(df['amount'].sum() / days_since_first)
    session_frequency = float(txn_count / days_since_first)
    avg_transaction_amount = float(df['amount'].mean())
    amount_var = float(df['amount'].var()) if txn_count > 1 else 0.0

    night_ratio = float(df['timestamp'].dt.hour.between(0, 5).sum() / txn_count)
    weekend_ratio = float(df['timestamp'].dt.dayofweek.isin([5, 6]).sum() / txn_count)
    upi_ratio = float((df['type'].str.lower() == 'upi').sum() / txn_count)
    trading_ratio = float((df['type'].str.lower() == 'trading').sum() / txn_count)

    feat_dict = {f: 0.0 for f in cache.feature_names}
    live_features = {
        'spending_velocity': spending_velocity,
        'session_frequency': session_frequency,
        'avg_transaction_amount': avg_transaction_amount,
        'amount_var': amount_var,
        'night_session_ratio': night_ratio,
        'weekend_ratio': weekend_ratio,
        'upi_ratio': upi_ratio,
        'trading_ratio': trading_ratio,
        'days_since_first_transaction': days_since_first,
        'total_transactions': float(txn_count)
    }
    feat_dict.update({k: v for k, v in live_features.items() if k in feat_dict})

    raw_vector = np.array([[feat_dict[f] for f in cache.feature_names]])
    scaled_vector = cache.scaler.transform(raw_vector)
    tensor_features = torch.tensor(scaled_vector, dtype=torch.float32)

    with torch.no_grad():
        preds = cache.model(tensor_features)
        probs = torch.softmax(preds, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(preds, dim=1).item()

    # SHAP explanation
    top_features = []
    try:
        shap_values = cache.explainer.shap_values(tensor_features)
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class][0]
        else:
            sv = shap_values[0, :, predicted_class]
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        top_features = [
            {"feature": cache.feature_names[i], "impact": round(float(sv[i]), 4)}
            for i in top_idx
        ]
    except Exception:
        pass

    state_names = {0: "Casual", 1: "Frequent", 2: "Compulsive", 3: "Crisis"}
    return {
        "user_id": payload.user_id,
        "user_risk_tier": predicted_class,
        "label": state_names[predicted_class],
        "probabilities": {state_names[i]: round(probs[i], 4) for i in range(4)},
        "top_features": top_features,
        "behavioral_state": state_names[predicted_class]
    }