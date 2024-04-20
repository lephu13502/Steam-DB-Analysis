import random
from collections import defaultdict

import pandas as pd
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SlopeOne, accuracy

import warnings



class CollaborativeFiltering():
    
    def __init__(self, df_file: str = "", model_classes: list = []) -> None:

        if len(df_file) > 0:
            self.read_csv(df_file)
        if len(model_classes) > 0:
            self.fit_models(model_classes)
            
    

    def read_csv(self, df_file):
        df = pd.read_csv(df_file)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df["timestamp_created"] = pd.to_datetime(df["timestamp_created"])
        df["timestamp_updated"] = pd.to_datetime(df["timestamp_updated"])
        df["last_played"] = pd.to_datetime(df["last_played"])
        self.df = df
        self.prepare()

    

    def prepare(self):
            # Add rating table for extraction
        self.rating_table = self.df[["steamid", "appid", "voted_up"]]
        self.rating_table["voted_up"] = self.rating_table["voted_up"].apply(lambda x: 1 if x else -1)
        user_activating = self.rating_table.groupby("steamid")["voted_up"] \
                                           .count() \
                                           .reset_index() \
                                           .rename(columns={ "voted_up": "review_count" })
        item_activating = self.rating_table.groupby("appid")["voted_up"] \
                                           .count() \
                                           .reset_index() \
                                           .rename(columns={ "voted_up": "review_count" })
        self.active_users = user_activating.loc[user_activating["review_count"] > 3, "steamid"]
        self.active_items = item_activating.loc[item_activating["review_count"] > 3, "appid"]
        self.rating_table = self.rating_table[
            self.rating_table["steamid"].isin(self.active_users) &
            self.rating_table["appid"].isin(self.active_items)
        ]
    

    
    def fit_models(self, model_classes=[]):

        # Shuffle the table before applying KFold
        trained_table = self.rating_table.sample(frac=1)
        trained_table.sample(frac=1)
        
        s_reader = Reader(rating_scale=(-1, 1))
        s_data = Dataset.load_from_df(trained_table, s_reader)
        trainset, testset = train_test_split(s_data, test_size=0.2)

        # Train models
        self.models = model_classes
        self.predictions = [None for _ in range(len(model_classes))]
        for i in range(len(model_classes)):
            self.models[i].fit(trainset)
            self.predictions[i] = self.models[i].test(testset)
            print(accuracy.rmse(self.predictions[i], verbose=True), '\n')



    def get_random_user(self, active_only: bool = True):
        return random.choice(self.rating_table["steamid"].unique() if active_only else self.df["steamid"].unique())


    
    def get_random_item(self, active_only: bool = True):
        return random.choice(self.rating_table["appid"].unique() if active_only else self.df["appid"].unique())

    
    
    # Get top 10 based on the models
    def get_top_10(self, user: int,
                   model_id_list: list, df_game: pd, df_bridge: pd,
                   top_n: int = 10, filter_func_list: list = [], ignored_appids: list = []):
    	
        tuple_list = defaultdict(lambda: -2)   # Rating range (-1, 1)
        for i in model_id_list:
            for uid, iid, true_r, est, _ in self.predictions[i]:
                # Find user in the whole predictions
                if uid == user:
                    tuple_list[iid] = max(tuple_list[iid], est)
                    
        # It's possible to get less result. In this case, use a brute-force method
        for item in self.active_items:
            tuple_list[item] = max(tuple_list[iid], self.models[i].predict(user, item).est)
    
        tuple_list = [(k, v) for k, v in tuple_list.items()]
        tuple_list.sort(key=lambda x: x[1], reverse=True)
    
        # Change the rating prediction
        recommendation_table = pd.DataFrame(tuple_list, columns=["appid", "likely_to_like"])
        recommendation_table["likely_to_like"] = recommendation_table["likely_to_like"].apply(
            lambda x: "YES" if x > 0.25 else ("NO" if x < -0.25 else "MAYBE"))
        recommendation_table["l"] = recommendation_table["likely_to_like"].apply(
            lambda x: 3 if (x == "YES") else (1 if (x == "NO") else 2))

        recommendation_table = recommendation_table[~recommendation_table["appid"].isin(ignored_appids)]
        
        recommendation_table = pd.merge(
            recommendation_table, df_game,
            how="left", left_on="appid", right_on="steam_appid")
        
        recommendation_table = pd.merge(
            recommendation_table, df_bridge,
            how="left", on="appid")

        # Apply filter
        if len(filter_func_list) > 0:
            for func in filter_func_list:
                recommendation_table = func(recommendation_table)
    
        # Drop all negative rating games
        # recommendation_table = recommendation_table[~recommendation_table["review_score_desc"].str.contains("Negative")]
        # Sort recommendations by order
        recommendation_table.sort_values(by=["l", "review_score"], inplace=True, ascending=False)
        # Remove unusued features
        recommendation_table.drop(columns=["l"], inplace=True)
        # Extract top N
        return recommendation_table[:top_n]
    


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    
    review = CollaborativeFiltering()
    review.read_csv("../../archive/cleaned_reviews_v2.csv")
    review.fit_models([SlopeOne()])


