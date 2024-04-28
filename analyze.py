import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
path_to_features = 'data/extracted_features.csv'
path_to_info = 'data/Music Info.csv'
def process_tags(tag_string):
    if pd.isna(tag_string) or tag_string.strip() == '':
        return []
    else:
        return tag_string.split(', ')

def classify_tags(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")
    return accuracy, y_pred


#used for df apply
def reduce_tags(df):
    def create_inverted_mapping(reduced_tags):
        inverted_mapping = {}
        for category, tags in reduced_tags.items():
            for tag in tags:
                inverted_mapping[tag] = category
        return inverted_mapping

    def reduce_tags(tags, inverted_mapping):
        reduced_categories = set()
        for tag in tags:
            if tag in inverted_mapping:
                reduced_categories.add(inverted_mapping[tag])
        return reduced_categories
    
    
    reduced_tags = {
        'rock': {'classic_rock', 'hard_rock', 'indie_rock', 'alternative_rock', 'punk_rock', 'post_punk', 'grunge', 'psychedelic_rock', 'progressive_rock', 'gothic', 'britpop', 'new_wave'},
        'metal': {'heavy_metal', 'black_metal', 'death_metal', 'thrash_metal', 'doom_metal', 'power_metal', 'metalcore', 'nu_metal', 'gothic_metal', 'progressive_metal', 'symphonic_metal', 'melodic_death_metal', 'grindcore'},
        'electronic': {'trance', 'electro', 'techno', 'house', 'synthpop', 'ambient', 'idm', 'downtempo', 'drum_and_bass', 'trip_hop', 'chillout', 'dark_ambient', 'industrial'},
        'pop': {'pop_rock', 'indie_pop', 'j_pop', 'british', 'american', 'swedish', 'french', 'german', 'polish', 'japanese', 'pop'},
        'hip_hop_rap': {'hip_hop', 'rap'},
        'jazz_blues_soul': {'jazz', 'blues', 'blues_rock', 'soul', 'funk'},
        'folk_country': {'folk', 'country', 'singer_songwriter'},
        'classical_new_age': {'classical', 'new_age'},
        'other_genres': {'punk', 'ska', 'reggae', 'emo', 'screamo', 'post_hardcore', 'hardcore', 'experimental', 'acoustic', 'cover', 'instrumental', 'noise', 'avant_garde', 'mellow', 'beautiful', 'love'},
        'decades': {'60s', '70s', '80s', '90s', '00s'},
        'global': {'russian', 'french', 'swedish', 'american', 'polish', 'german', 'british', 'japanese'}
    }

    inverted_mapping = create_inverted_mapping(reduced_tags)

    df['tags'] = df['tags'].apply(lambda tags: reduce_tags(tags, inverted_mapping))

    return df

    





def main():
    df = pd.read_csv(path_to_features)
    df.set_index('track_id', inplace=True) # Set the track_id as the index
    music_info = pd.read_csv(path_to_info)
    labels = music_info.copy()
    labels.set_index('track_id', inplace=True) # Set the track_id as the index
    print(df.head())
    print(labels.head())

    df = df.join(labels)
    print(df.head())
    print(df.shape[0])

    # df['tags'] = df['tags'].apply(process_tags)
    # df = reduce_tags(df)
    #print disitnct tags
    # all_tags = set()
    # for tags in df['tags']:
    #     all_tags.update(tags)
    # mlb = MultiLabelBinarizer()

    # encoded_labels = mlb.fit_transform(df['genre'])
    # print(encoded_labels[:5])
   
    # features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    # features = ['danceability','speechiness', 'acousticness', 'liveness', 'valence', 'tempo']
    # X = df[features]
    # y= df['genre']  
    # cols = X.columns
    # min_max_scaler = preprocessing.MinMaxScaler()
    # np_scaled = min_max_scaler.fit_transform(X)

    # new data frame with the new scaled data. 
    # X = pd.DataFrame(np_scaled, columns = cols)

     #take columns with word mean as features
    features = [col for col in df.columns if 'mean' in col]
    X = df[features]
    y= df['genre']  
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns = cols)

    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.show()
        #index the genre y
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
    print(X_train[:5])

    xgb_cls = XGBClassifier(n_estimators=30, learning_rate=0.05)
    rforest = RandomForestClassifier(n_estimators=10000, max_depth=10, random_state=0)
    
    knn = KNeighborsClassifier(n_neighbors=19)


    # Random Forest
    rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)

    # Support Vector Machine
    svm = SVC(decision_function_shape="ovo")


    accuracy, y_pred = classify_tags(xgb_cls, X_train, y_train, X_train, y_train)



if __name__ == '__main__':
    main()