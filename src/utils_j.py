from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

def generate_validation_set(session, tracking, test_size=0.1, seed=42):
    '''
    trn_session, trn_tracking, val_session, val_tracking = generate_validation_set(train_session, 
                                                                train_tracking, test_size=0.1, seed=42)
    '''
    trn_session, val_session = train_test_split(session, test_size=test_size, random_state=seed, shuffle=True)
    trn_session_ids, val_session_ids = set(trn_session.sid), set(val_session.sid)
    trn_tracking = tracking[tracking.sid.apply(lambda x: x in trn_session_ids)]
    val_tracking = tracking[tracking.sid.apply(lambda x: x in val_session_ids)]
    return trn_session, trn_tracking, val_session, val_tracking

def print_results(trn_logloss, val_logloss):
    print(f'Train logloss: {100*np.mean(trn_logloss):.2f} +/- {200*np.std(trn_logloss):.2f} | '
          f'Val logloss: {100*np.mean(val_logloss):.2f} +/- {200*np.std(val_logloss):.2f}')
    

def test_lgbm(lgbm, params, X, y, X_test, test_size, random_seeds = [42], cat_features=[]):
    trn_logloss, val_logloss = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    idx = 0
    for random_seed in random_seeds:
        X_train, X_val, y_train, y_val  = train_test_split(X, 
                                                           y, 
                                                           test_size=test_size, 
                                                           random_state=random_seed, 
                                                           shuffle=True)
    
        num_rounds = 10000
        d_train = lgbm.Dataset(X_train, y_train)
        d_valid = lgbm.Dataset(X_val, y_val)
        bst = lgbm.train(params, d_train, 
                         num_rounds, [
                             d_valid], 
                         early_stopping_rounds=30, 
                         # categorical_features = cat_features, 
                         verbose_eval=False)
        y_trn_pred = bst.predict(X_train)
        y_val_pred = bst.predict(X_val)
        trn_logloss.append(log_loss(y_train, y_trn_pred))
        val_logloss.append(log_loss(y_val, y_val_pred))
        print(f'No. estimators: {bst.best_iteration} | '
                  f'Train log loss: {trn_logloss[-1]} | '
                  f'Val log loss: {val_logloss[-1]}')
        
        y_tst_pred = bst.predict(X_test)
        y_test += y_tst_pred
        y_pred = y_val_pred
        idx += 1
    print()
    print_results(trn_logloss, val_logloss)
    print()
    return y_test / len(random_seeds), y_pred

def test_catboost(model, X, y, X_test, test_size, random_seeds = [42], cat_features=[]):
    trn_logloss, val_logloss = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    for random_seed in random_seeds:
        X_trn, X_val, y_trn, y_val  = train_test_split(X, 
                                                           y, 
                                                           test_size=test_size, 
                                                           random_state=random_seed, 
                                                           shuffle=True)
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)],
                  use_best_model=True, cat_features=cat_features, 
                  verbose=False, early_stopping_rounds=30)
        y_trn_pred = model.predict_proba(X_trn)[:,1]
        y_val_pred = model.predict_proba(X_val)[:,1]
        trn_logloss.append(log_loss(y_trn, y_trn_pred))
        val_logloss.append(log_loss(y_val, y_val_pred))
        print(f'No. estimators: {model.tree_count_} | '
                  f'Train log loss: {trn_logloss[-1]} | '
                  f'Val log loss: {val_logloss[-1]}')
        
        y_tst_pred = model.predict_proba(X_test)[:,1]
        y_test += y_tst_pred
        
    print()
    print_results(trn_logloss, val_logloss)
    print()
    return y_test / len(random_seeds), y_pred
