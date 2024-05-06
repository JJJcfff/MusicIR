import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)  # Ensure there are no missing values
    return df


def preprocess_features(df, features, feature_weights):
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_weighted = X_scaled * feature_weights  # Apply weights

    X_normalized_per_feature = normalize(X_scaled_weighted, axis=0)

    return X_scaled_weighted, X_normalized_per_feature


def define_feature_weights(features):
    weights = np.ones(len(features))  # Start with equal weights
    individual_weights = pd.read_csv('individual_feature_results.csv')
    ranking = individual_weights.groupby('Feature')['Accuracy'].max().sort_values(ascending=False)
    print(ranking)
    baseline_accuracy = 1.0/15.0  # 1/number of classes
    for f in features:

        acc = ranking[f]
        weights[features.index(f)] = (acc-baseline_accuracy) / baseline_accuracy

        print(f'{f}: {acc:.5f} ({weights[features.index(f)]:.5f})')

    return weights


def train_classifiers(X_train, y_train, X_test, y_test, print_results=False):
    classifiers = {
        'KNN Euclidean': KNeighborsClassifier(n_neighbors=5),
        'KNN Cosine': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVC': SVC(kernel='linear'),
        'XGB': XGBClassifier(n_estimators=30, learning_rate=0.05)
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if print_results:
            print(f'{name} accuracy: {accuracy:.2f}')
    return results


def main():
    df = load_data('data/extracted_features.csv')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['genre'])

    print("Classifying genres based on individual features:")
    features = list(df.columns[3:])
    feature_results = []

    for f in tqdm(features, desc='Features'):
        X_scaled_weighted, X_normalized = preprocess_features(df, [f], [1])
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled_weighted, y, test_size=0.2,
                                                                          random_state=42)
        X_train_norm, X_test_norm, _, _ = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

        scaled_results = train_classifiers(X_train_scaled, y_train, X_test_scaled, y_test)
        norm_results = train_classifiers(X_train_norm, y_train, X_test_norm, y_test)

        for classifier, acc in scaled_results.items():
            feature_results.append({'Feature': f, 'Classifier': classifier, 'Accuracy': acc, 'Normalization': 'Scaled'})
        for classifier, acc in norm_results.items():
            feature_results.append(
                {'Feature': f, 'Classifier': classifier, 'Accuracy': acc, 'Normalization': 'Normalized'})

    feature_results_df = pd.DataFrame(feature_results)
    feature_results_df.to_csv('individual_feature_results.csv', index=False)

    print("\nClassifying genres based on all features:")

    feature_weights = define_feature_weights(features)

    X_scaled_weighted, X_normalized = preprocess_features(df, features, feature_weights)

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled_weighted, y, test_size=0.2,
                                                                      random_state=42)
    X_train_norm, X_test_norm, _, _ = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    print("Results using scaled and weighted features:")
    scaled_results = train_classifiers(X_train_scaled, y_train, X_test_scaled, y_test, print_results=True)

    print("\nResults using normalized features:")
    norm_results = train_classifiers(X_train_norm, y_train, X_test_norm, y_test, print_results=True)


if __name__ == '__main__':
    main()
