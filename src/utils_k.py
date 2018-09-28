import pandas as pd
import numpy as np

def add_type_features(df, all_data):
    """
        Agrega las cuentas normalizadas 
    """
        
    temp = all_data.groupby(['sid', 'type']).count().reset_index()[['sid', 'type', 'duration']]
    temp_pivot = temp.pivot_table(values='duration', index=temp.sid, columns='type', aggfunc='first').reset_index()
    temp_pivot['SUM_TYPE'] = temp_pivot.sum(axis=1)
    temp_pivot.iloc[:, 1:-1] = temp_pivot.iloc[:, 1:-1].div(temp_pivot['SUM_TYPE'], axis=0)
    temp_pivot.columns = [str(col) + '_type' if col != 'sid' else str(col) for col in temp_pivot.columns]
    df = pd.merge(df, temp_pivot, on='sid')
    
    temp = all_data.groupby(['sid', 'type_simplified']).count().reset_index()[['sid', 'type_simplified', 'duration']]
    temp_pivot = temp.pivot_table(values='duration', index=temp.sid, columns='type_simplified', aggfunc='first').reset_index()
    temp_pivot['SUM_TYPE_SIMPLIFIED'] = temp_pivot.sum(axis=1)
    temp_pivot.iloc[:, 1:-1] = temp_pivot.iloc[:, 1:-1].div(temp_pivot['SUM_TYPE_SIMPLIFIED'], axis=0)
    temp_pivot.columns = [str(col) + '_sim_type' if col != 'sid' else str(col) for col in temp_pivot.columns]
    df = pd.merge(df, temp_pivot, on='sid')
    
    df.fillna(0, inplace=True)
    return df