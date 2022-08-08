import os
import cudf
import pandas as pd 


DATETIME_CONVERTION = 'ms'

# filter out categories with features coverage more than 80%, this helped in keeping quality features
def process_item_features(DATA_FOLDER, category_coverage_min=0.8):
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'item_features.csv'))
    tmp = df.feature_category_id.value_counts()/df.item_id.nunique()
    categories_to_keep = [3,  4,  5, 17, 24, 30, 45, 46, 53, 55, 58, 63, 65, 73]
    categories_to_keep = list(set(categories_to_keep + tmp[tmp>=category_coverage_min].index.tolist()))
    df = df[df.feature_category_id.isin(categories_to_keep)]
    df = df[~df.feature_category_id.isin([[30, 4, 46, 28, 53, 1]])]
    df = df.pivot_table('feature_value_id', ['item_id'], 'feature_category_id').reset_index()
    df.columns = [str(col)+'_f' if isinstance(col, int) else str(col) for col in df.columns]
#     df.columns = [str(col) for col in df.columns]
    return df

# add timestamp and day features
def process_date_column(ddf):
    ddf['date'] = ddf['date'].astype(f'datetime64[{DATETIME_CONVERTION}]')
    ddf['timestamp'] = ddf['date'].astype('int64')
    ddf = ddf.sort_values(['session_id', 'date']).reset_index(drop=True)
    ddf['day'] = (ddf['date'] - ddf['date'].min()).dt.days
    return ddf

def get_drepessi_recsys2022_dataset(input_path):
    DATA_FOLDER = input_path
    # get the item features
    item_features = cudf.from_pandas(process_item_features(DATA_FOLDER))

    # load data
    sessions = cudf.read_csv(os.path.join(DATA_FOLDER, 'train_sessions.csv'))
    purchases = cudf.read_csv(os.path.join(DATA_FOLDER, 'train_purchases.csv'))

    # merge session data with item features 
    sessions = cudf.merge(sessions, item_features, on='item_id', how='left')
    purchases = cudf.merge(purchases, item_features, on='item_id', how='left')

    # add timestamp and day features, and convert the format of date to ms
    sessions = process_date_column(sessions)
    purchases = process_date_column(purchases)
    purchases = purchases.rename(columns={"item_id": "purchase_id"})

    # Split into train and validation set
    train_session = sessions.loc[sessions.day <= (sessions.day.max()-30) ].copy().reset_index(drop=True)
    valid_session = sessions.loc[sessions.day > (sessions.day.max()-30) ].copy().reset_index(drop=True)
    train_session.shape, valid_session.shape

    # Merge with train + valid purchases
    train_purchases = purchases[purchases.session_id.isin(train_session.session_id.unique().values.tolist())]
    valid_purchases = purchases[purchases.session_id.isin(valid_session.session_id.unique().values.tolist())]

    train_purchases = train_purchases[['session_id','purchase_id']]
    train = cudf.merge(train_session, train_purchases, on='session_id', how='left')

    valid_purchases = valid_purchases[['session_id','purchase_id']]
    valid = cudf.merge(valid_session, valid_purchases, on='session_id', how='left')
    
    purchases = purchases[['session_id','purchase_id']]
    sessions = cudf.merge(sessions, purchases, on='session_id', how='left')
    
    train = train.fillna(-1)
    valid = valid.fillna(-1)
    sessions = sessions.fillna(-1)
    return train, valid, sessions
