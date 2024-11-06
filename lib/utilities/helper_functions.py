import warnings
import numpy as np
import pandas  as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import yaml
from pandas_profiling import ProfileReport


class HelperFunctions :

    def __init__(self):
        pass


    def encode_features(self, df, encoding_methods=None, subset=None, one_hot_threshold=2799,
                        custom_mappings=None, global_encoding_method=None):
        """
        Encodes categorical features in a dataframe with multiple encoding methods or a single method for all.

        Parameters:
            df (pd.DataFrame): The dataframe to encode.
            encoding_methods (dict, optional): Dictionary mapping feature names to encoding methods.
                                                Available methods: 'one-hot', 'label', 'ordinal', 'custom'.
            subset (list, optional): A list of features to encode. If None, all categorical features will be encoded.
            one_hot_threshold (int, optional): Maximum number of unique categories to encode using one-hot encoding.
                                                If the number of unique categories in a feature exceeds this value,
                                                the feature will not be one-hot encoded.
            custom_mappings (dict, optional): Dictionary mapping feature names to custom encoding mappings.
                                                e.g., {'feature1': {value1: key1, value2: key2}, ...}
            global_encoding_method (str, optional): Single encoding method to apply to all categorical features.
                                                     Overrides encoding_methods if provided.

        Returns:
            pd.DataFrame: The dataframe with encoded features.
        """

        df_encoded = df.copy()

        # Select subset of features to encode
        if subset is None:
            # If subset not provided, select all categorical features
            subset = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

        for feature in subset:
            # Determine the encoding method to use
            if encoding_methods and feature in encoding_methods:
                method = encoding_methods[feature]
            else:
                method = global_encoding_method

            if method == 'one-hot':
                # Check if feature exceeds one-hot encoding threshold
                num_unique_values = df_encoded[feature].nunique()
                if one_hot_threshold and num_unique_values > one_hot_threshold:
                    print(f"Skipping one-hot encoding for {feature} due to threshold")
                    continue
                # Perform one-hot encoding (keep all categories) and convert to integers
                one_hot_encoded = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=False).astype(int)
                df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1).drop(columns=[feature])

            elif method == 'label':
                # Perform label encoding
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))

            elif method == 'ordinal':
                # Perform ordinal encoding
                oe = OrdinalEncoder()
                df_encoded[feature] = oe.fit_transform(df_encoded[[feature]]).astype(int)

            elif method == 'custom' and custom_mappings and feature in custom_mappings:
                # Apply custom mapping
                df_encoded[feature] = df_encoded[feature].map(custom_mappings[feature])

            elif method is None:
                print(f"No encoding method specified for {feature}.")
            else:
                print(f"Encoding method for {feature} not recognized or not provided.")

        return df_encoded

    def scale_dataframe(self, df, method='minmax', features=None):
        """
        Scales the numerical data in a Pandas DataFrame using either Min-Max Scaling or Z-Score Normalization.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be scaled.
        method (str): The scaling method to use. Choose 'minmax' or 'zscore'. Default is 'minmax'.
        features (list): Optional. List of specific features/columns to scale. If None, scales all numerical columns.

        Returns:
        pd.DataFrame: The scaled DataFrame, with only the numerical features scaled.
        """
        # Step 1: Check if there are any numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns
        if len(numerical_cols) == 0:
            raise ValueError("No numerical features found in the dataset to scale.")

        # Step 2: If no specific features provided, use all numerical features
        if features is None:
            features = numerical_cols
        else:
            # Ensure the selected features are numerical
            features = [col for col in features if col in numerical_cols]

            if len(features) == 0:
                raise ValueError("No numerical features found in the specified subset of columns.")

        # Step 3: Create a copy of the DataFrame to avoid modifying the original one
        df_scaled = df.copy()

        # Step 4: Apply scaling to the selected numerical features
        if method == 'minmax':
            # Min-Max Scaling
            df_scaled[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())

        elif method == 'zscore':
            # Z-Score Normalization
            df_scaled[features] = (df[features] - df[features].mean()) / df[features].std()

        else:
            raise ValueError("Method must be either 'minmax' or 'zscore'")

        return df_scaled

    def map_features(self, df, custom_mappings=None):
        """
        Apply custom mappings to specified categorical features.

        Parameters:
            df (pd.DataFrame): The dataframe containing the features to be mapped.
            custom_mappings (dict or None): Dictionary mapping feature names to custom encoding mappings.
                                            e.g., {'feature1': {value1: key1, value2: key2}, ...}.
                                            If None, no mapping will be applied.

        Returns:
            pd.DataFrame: The dataframe with mapped features or the original dataframe if no mapping is provided.
        """
        # If custom_mappings is None, return the original DataFrame without modifications
        if custom_mappings is None:
            print("No custom mappings provided. Returning the original DataFrame.")
            return df

        # Make a copy of the DataFrame to avoid modifying the original
        df_mapped = df.copy()

        # Apply the mappings to the specified features
        for feature, mapping in custom_mappings.items():
            if feature in df_mapped.columns:
                # Use replace instead of map to avoid overwriting non-mapped values with NaN
                df_mapped[feature] = df_mapped[feature].replace(mapping)
            else:
                print(f"Feature '{feature}' not found in the DataFrame.")

        return df_mapped

    def data_imputer(self, df, num_impute_method='mean', cat_impute_method='most_frequent',
                     num_constant_value=None, cat_constant_value=None,
                     subset_numerical=None, subset_categorical=None,
                     custom_impute_methods=None):
        """
        Perform data imputation on a given DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - num_impute_method (str): Default imputation method for numerical features ('mean', 'median', 'constant').
        - cat_impute_method (str): Default imputation method for categorical features ('most_frequent', 'constant', 'drop', 'ffill', 'bfill').
        - num_constant_value (any): The constant value to use for numerical imputation (if applicable).
        - cat_constant_value (any): The constant value to use for categorical imputation (if applicable).
        - subset_numerical (list of str): List of numerical feature names to be considered for imputation (default: None).
        - subset_categorical (list of str): List of categorical feature names to be considered for imputation (default: None).
        - custom_impute_methods (dict): A dictionary specifying feature names as keys and imputation methods as values.

        Returns:
        - pd.DataFrame: DataFrame with imputed values.

        Raises:
        - ValueError: For various error scenarios including missing features, invalid methods, etc.
        """

        # Validate input DataFrame
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Select numerical and categorical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle subsets for numerical and categorical features
        if subset_numerical is not None:
            numerical_features = [feat for feat in subset_numerical if feat in numerical_features]
        if subset_categorical is not None:
            categorical_features = [feat for feat in subset_categorical if feat in categorical_features]

        # Error handling for missing features
        if not numerical_features and num_impute_method != 'drop':
            raise ValueError("No numerical features found for imputation.")
        if not categorical_features and cat_impute_method != 'drop':
            raise ValueError("No categorical features found for imputation.")

        # Impute numerical features
        for feature in numerical_features:
            # Determine the imputation method
            impute_method = custom_impute_methods.get(feature,
                                                      num_impute_method) if custom_impute_methods else num_impute_method

            if df[feature].isnull().all():
                if impute_method == 'constant':
                    if num_constant_value is None or not isinstance(num_constant_value, (int, float)):
                        raise ValueError(
                            "Constant value must be specified for constant imputation and must be a number.")
                    df[feature] = num_constant_value
                else:
                    raise ValueError(
                        f"All values in feature '{feature}' are null. Cannot apply '{impute_method}' imputation method.")
            else:
                if impute_method == 'mean':
                    df[feature] = df[feature].fillna(df[feature].mean())
                elif impute_method == 'median':
                    df[feature] = df[feature].fillna(df[feature].median())
                elif impute_method == 'constant':
                    if num_constant_value is None or not isinstance(num_constant_value, (int, float)):
                        raise ValueError(
                            "Constant value must be specified for constant imputation and must be a number.")
                    df[feature] = df[feature].fillna(num_constant_value)
                else:
                    raise ValueError(
                        f"Invalid numerical imputation method: {impute_method}. Choose from 'mean', 'median', 'constant'.")

        # Impute categorical features
        for feature in categorical_features:
            # Determine the imputation method
            impute_method = custom_impute_methods.get(feature,
                                                      cat_impute_method) if custom_impute_methods else cat_impute_method

            if df[feature].isnull().all():
                if impute_method == 'constant':
                    if cat_constant_value is None or not isinstance(cat_constant_value, str):
                        raise ValueError(
                            "Constant value must be specified for constant imputation and must be a string.")
                    df[feature] = cat_constant_value
                elif impute_method == 'drop':
                    df.drop(columns=[feature], inplace=True)
                else:
                    raise ValueError(
                        f"All values in feature '{feature}' are null. Cannot apply '{impute_method}' imputation method.")
            else:
                if impute_method == 'most_frequent':
                    df[feature] = df[feature].fillna(df[feature].mode()[0])
                elif impute_method == 'constant':
                    if cat_constant_value is None or not isinstance(cat_constant_value, str):
                        raise ValueError(
                            "Constant value must be specified for constant imputation and must be a string.")
                    df[feature] = df[feature].fillna(cat_constant_value)
                elif impute_method == 'drop':
                    df.dropna(subset=[feature], inplace=True)
                elif impute_method == 'ffill':
                    df[feature] = df[feature].fillna(method='ffill')
                elif impute_method == 'bfill':
                    df[feature] = df[feature].fillna(method='bfill')
                else:
                    raise ValueError(
                        f"Invalid categorical imputation method: {impute_method}. Choose from 'most_frequent', 'constant', 'drop', 'ffill', 'bfill'.")

        return df

    def convert_feature_types(self, df, feature_type_mapping=None):
        """
        Convert the data types of specific features in a DataFrame according to the given mapping.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the features.
            feature_type_mapping (dict or None): A dictionary mapping feature names to their desired data types.
                                                e.g., {'age': 'int', 'salary': 'float', 'signup_date': 'datetime64', ...}.
                                                If None, the original DataFrame will be returned.

        Returns:
            pd.DataFrame: A new DataFrame with the features converted to the specified types or the original DataFrame if no mapping is provided.
        """
        # If feature_type_mapping is None, return the original DataFrame without modifications
        if feature_type_mapping is None:
            print("No feature type mapping provided. Returning the original DataFrame.")
            return df

        # Make a copy of the DataFrame to avoid modifying the original
        df_converted = df.copy()

        # Iterate over the feature_type_mapping dictionary and convert each feature's data type
        for feature, dtype in feature_type_mapping.items():
            if feature in df_converted.columns:
                try:
                    # Convert the feature's data type
                    df_converted[feature] = df_converted[feature].astype(dtype)
                    print(f"Feature '{feature}' successfully converted to {dtype}.")
                except ValueError as e:
                    print(f"Error converting feature '{feature}' to {dtype}: {e}")
            else:
                print(f"Feature '{feature}' not found in the DataFrame.")

        return df_converted

    def generate_profile_report(self, data: pd.DataFrame, config_path: str, output_path: str = "output.html") :
        """
        Generates a profile report for the given dataset using configuration from a YAML file.

        Parameters:
        data : pd.DataFrame
            The input data for which the profile report is to be generated.
        config_path : str
            The path to the YAML configuration file.
        output_path : str, optional
            The path where the HTML report will be saved. Default is 'output.html'.

        Returns:
        None
        """

        # Load configuration from the YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Suppress all warnings to ensure a clean output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Generate profile report with configuration from YAML file
            profile = ProfileReport(data,
                                    vars=config.get('vars', None),
                                    missing_diagrams=config.get('missing_diagrams', True),
                                    correlations=config.get('correlations', {}),
                                    interactions=config.get('interactions', True),
                                    html=config.get('html', {}),
                                    minimal=config.get('minimal', False))

            # Export the generated profile report to an HTML file at the specified path
            profile.to_file(output_path)
