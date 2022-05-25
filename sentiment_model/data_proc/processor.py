import os
import sentiment_model
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sentiment_model import app_config
from sentiment_model.data_proc.normalizer import TextNormalizer

random_state = 42


def read_and_normalize_dataset(dataset_path):
    dataset_df = pd.read_csv(dataset_path)
    # print(dataset_df.sentiment.value_counts())
    dataset_df["sentiment"] = dataset_df.sentiment.replace(to_replace=['negative', 'positive'],
                                                           value=[0, 1])
    dataset_df['review'] = dataset_df['review'].apply(TextNormalizer().normalize)
    print(dataset_df.sentiment.value_counts())

    dataset_df.sentiment.value_counts().plot(kind='bar', title='Count (sentiment)')
    plt.savefig(app_config["main_data_plot_path"], dpi=300)
    return dataset_df


def random_over_and_under_sampling_pandas(df_for_sampling):
    count_sentiment_1, count_sentiment_0 = df_for_sampling.sentiment.value_counts()
    df_sentiment_0 = df_for_sampling[df_for_sampling["sentiment"] == 0]
    df_sentiment_1 = df_for_sampling[df_for_sampling["sentiment"] == 1]

    df_sentiment_1_under = df_sentiment_1.sample(count_sentiment_0, replace=True)
    df_test_under = pd.concat([df_sentiment_0, df_sentiment_1_under], axis=0)

    df_sentiment_0_over = df_sentiment_0.sample(count_sentiment_1, replace=True)
    df_test_over = pd.concat([df_sentiment_1, df_sentiment_0_over], axis=0)

    print('Random under-sampling:')
    print(df_test_under.sentiment.value_counts())
    print('Random over-sampling:')
    print(df_test_over.sentiment.value_counts())
    return df_test_under, df_test_over


def random_over_and_under_sampler_imblearn(df_for_sampling):
    x = df_for_sampling["review"].to_numpy()
    y = df_for_sampling["sentiment"].to_numpy()

    ros_over = RandomOverSampler(random_state=42)
    x_over, y_over = ros_over.fit_resample(df_for_sampling, y)
    df_over = pd.DataFrame(data=x_over, columns=["review"])
    df_over["sentiment"] = y_over

    ros_under = RandomUnderSampler(random_state=42)
    x_under, y_under = ros_under.fit_resample(df_for_sampling, y)
    df_under = pd.DataFrame(data=x_under, columns=["review"])
    df_under["sentiment"] = y_under

    print('Random over-sampling:')
    print(df_over.sentiment.value_counts())
    print('Random under-sampling:')
    print(df_under.sentiment.value_counts())

    return df_over, df_under


def split_train_val_test(normalized_df):
    train_df, val_df = train_test_split(normalized_df, test_size=0.1)
    train_df, test_df = train_test_split(normalized_df, test_size=0.1)
    return train_df, val_df, test_df


if __name__ == "__main__":
    abs_dir_path = os.path.dirname(os.path.abspath(sentiment_model.__file__))

    path_of_dataset = os.path.join(abs_dir_path, app_config['main_dataset'])

    path_of_train = os.path.join(abs_dir_path, app_config['train_data_path'])
    path_of_val = os.path.join(abs_dir_path, app_config['val_data_path'])
    path_of_test = os.path.join(abs_dir_path, app_config['test_data_path'])

    path_of_over_sampled_train = os.path.join(abs_dir_path, app_config['train_over_sampled_data_path'])
    path_of_over_sampled_val = os.path.join(abs_dir_path, app_config['val_over_sampled_data_path'])
    path_of_over_sampled_test = os.path.join(abs_dir_path, app_config['test_over_sampled_data_path'])

    path_of_under_sampled_train = os.path.join(abs_dir_path, app_config['train_under_sampled_data_path'])
    path_of_under_sampled_val = os.path.join(abs_dir_path, app_config['val_under_sampled_data_path'])
    path_of_under_sampled_test = os.path.join(abs_dir_path, app_config['test_under_sampled_data_path'])

    norm_df = read_and_normalize_dataset(path_of_dataset)
    resampled_df_over, resampled_df_under = random_over_and_under_sampler_imblearn(norm_df)

    train_norm, val_norm, test_norm = split_train_val_test(norm_df)
    train_under, val_under, test_under = split_train_val_test(resampled_df_under)
    train_over, val_over, test_over = split_train_val_test(resampled_df_over)

    train_norm.to_csv(path_of_train, index=False)
    val_norm.to_csv(path_of_val, index=False)
    test_norm.to_csv(path_of_test, index=False)

    train_under.to_csv(path_of_under_sampled_train, index=False)
    val_under.to_csv(path_of_under_sampled_val, index=False)
    test_under.to_csv(path_of_under_sampled_test, index=False)

    train_over.to_csv(path_of_over_sampled_train, index=False)
    val_over.to_csv(path_of_over_sampled_val, index=False)
    test_over.to_csv(path_of_over_sampled_test, index=False)
