import re
import json
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity



class ContentBasedFiltering():
    
    
    def __init__(self, df_file: str = "") -> None:
        if len(df_file) > 0:
            self.read_csv(df_file)

    
    
    def read_csv(self, df_file: str):
        self.df = pd.read_csv("archive/cleaned_steam_db_v2.csv")
        self.df["genres"] = self.df["genres"].apply(json.loads)
        self.df["developers"] = self.df["developers"].apply(json.loads)
        self.df["publishers"] = self.df["publishers"].apply(json.loads)

    
    
    def get_genre_dict(self, genre_dict_file: str):
        self.genre_dict = {}
        with open(genre_dict_file, "r") as f:
        	self.genre_dict = json.loads(f.read())
        f.close()


    
    def get_studio_dict(self, studio_dict_file: str):
        self.studio_dict = {}
        with open(studio_dict_file, "r", encoding="utf-8") as f:
        	self.studio_dict = json.loads(f.read())
        f.close()



    # ! Use it with caution: Not run yet
    def train_models(self, model_class_funcs=[], gensim_file_links=[], vector_file_links=[], english_only=True):
        
        global count
    
        if english_only:
            df_extract = self.df.loc[df_game["lang_en"], [*lang_key, 'type', 'developers', 'publishers', 'genres',
                                                          'description', 'tool', 'nsfw', 'film', 'steam_appid']].reset_index(drop=True)
        else:
            df_extract = self.df[[*lang_key, 'type', 'developers', 'publishers', 'genres',
                                  'description', 'tool', 'nsfw', 'film', 'steam_appid']].reset_index(drop=True)
    
        df_extract["description"].fillna("", inplace=True)
        
        # Make a pre-embedding model
        print("Start cleaning description")
        count = 0
        df_extract["pre_word_embedding"] = \
            df_extract["description"].apply(lambda x: to_simplified_sentence_list(x))
        
        print("Add type, developers, and publishers")
        count = 0
        df_extract["pre_word_embedding"] = \
            df_extract[["pre_word_embedding", "type", "developers", "publishers"]].apply(add_dev_n_pub, axis=1)
    
        print("Add genres")
        count = 0
        df_extract["pre_word_embedding"] = \
            df_extract[["pre_word_embedding", "genres"]].apply(add_genres, axis=1)
    
        print("Add languages")
        count = 0
        df_extract["pre_word_embedding"] = \
            df_extract[["pre_word_embedding", *lang_key]].apply(add_language, axis=1)
    
        df_extract = df_extract
    
        # Add models
        self.models = [None for _ in range(len(model_class_funcs))]
        self.id_to_vector_dicts = [{} for i in range(len(model_class_funcs))]
        
        for i in range(len(model_class_funcs)):
            self.models[i] = model_class_funcs(df_extract)
            self.models[i].save(gensim_file_links[i])
            print(f"Finish model {i}")
            
            for I in range(self.df_english.shape[0]):
                self.id_to_vector_dicts[i][df_extract.loc[I, "steam_appid"]] = \
                    get_average_vector(df_extract.loc[I, "pre_word_embedding"], self.models[i])
            np.save(vector_file_links[i], self.id_to_vector_dicts[i])
        
        
    # Imagine the model is already trained
    def load_models(self, gensim_file_links: list = [], model_classes = [], vector_file_links: list = []):
        self.models = [None for _ in range(len(gensim_file_links))]
        self.id_to_vector_dicts = [{} for i in range(len(gensim_file_links))]
        for i in range(len(gensim_file_links)):
            self.models[i] = model_classes[i].load(gensim_file_links[i])
            with open(vector_file_links[i], "r") as f:
                self.id_to_vector_dicts[i] = json.loads(f.read())
            f.close()


    
    def get_similar_texts(self, texts: list, model_id: int, size: int = 10):
        word_table = pd.concat([
            pd.DataFrame(
                self.models[model_id].wv.most_similar(texts[i], topn=size),
                columns=[f"'{texts[i]}' similar text", f"'{texts[i]}' cos-sim"]
            ) for i in range(len(texts))
        ], axis=1)
        word_table.loc[:, word_table.columns.str.contains("cos-sim")] = \
            word_table.loc[:, word_table.columns.str.contains("cos-sim")].apply(lambda x: round(x, 3))
    
        return word_table


    
    def compare_two_games(self, name_1: str, name_2: str, printing: bool = True):
        
        labels = ["type", "name", "genres", "developers", "steam_appid", "lang_en", "lang_zh", "description"]
        
        try:
            id_1 = self.df.loc[self.df["name"] == name_1, "steam_appid"].values[0]
        except:
            print(f"Unfortunately, this game {name_1} maybe has a typo or does not exist in this dataframe.")
            return None, None
        try:
            id_2 = self.df.loc[self.df["name"] == name_2, "steam_appid"].values[0]
        except:
            print(f"Unfortunately, this game {name_2} maybe has a typo or does not exist in this dataframe.")
            return None, None
            
        compare_table = pd.concat([
            self.df.loc[self.df["steam_appid"] == id_1, labels],
            self.df.loc[self.df["steam_appid"] == id_2, labels]
        ], axis=0)

        cos_sim_list = [0 for _ in range(len(self.models))]
        for i in range(len(self.models)):
            cos_sim_list[i] = cosine_similarity([self.id_to_vector_dicts[i][str(id_1)]],
                                                [self.id_to_vector_dicts[i][str(id_2)]])[0][0]
            if printing:
                print("Model {}: {}".format(i, cos_sim_list[i]))   

        return compare_table, cos_sim_list


    
    def get_top_10(self, name: str = None, appid: int = None, vectorised_id: int = None,
                   df_bridge = None, top_n: int = 10, filter_func_list: list = [],
                   ignored_appids: list = []):

        global df_game, df_game_english
        
        if vectorised_id is None:
            raise Exception("Please add a ID converter to vectors")

        # Check validation
        if appid is None:
            if len(name) == 0:
                raise Exception("Please add name or appid")
            appid = self.df.loc[self.df["name"] == name, "steam_appid"].values[0]
        
        # Get embedded vector of the game
        try:  # Maybe the input is not an English game (or data bug, see more below)
            curr_vector = self.id_to_vector_dicts[vectorised_id][str(appid)]
        except:
            print("Unfortunately, this game cannot be recommended since it doesn't have English language or the dataframe about this game is buggy.")
        
        # Calculate the cosine similarities for each game
        curr_id_to_vector = self.id_to_vector_dicts[vectorised_id]
        tuple_list = [None for _ in curr_id_to_vector.keys()]
        for i, (k, v) in enumerate(curr_id_to_vector.items()):
            tuple_list[i] = (k, cosine_similarity([curr_vector], [v])[0][0])

        tuple_list.sort(key=lambda x: x[1], reverse=True)
        recommendation_table = pd.DataFrame(tuple_list, columns=["appid", "likely_to_like"])
        recommendation_table["appid"] = recommendation_table["appid"].astype(int)

        recommendation_table = recommendation_table[recommendation_table["appid"] != appid]
        recommendation_table = recommendation_table[~recommendation_table["appid"].isin(ignored_appids)]
        
        recommendation_table = pd.merge(
            recommendation_table, self.df,
            how="left", left_on="appid", right_on="steam_appid")

        recommendation_table = pd.merge(
            recommendation_table, df_bridge,
            how="left", on="appid")

        # Apply filter
        if len(filter_func_list) > 0:
            for func in filter_func_list:
                recommendation_table = func(recommendation_table)

        recommendation_table.drop(columns=["likely_to_like"], inplace=True)
        
        return recommendation_table[:top_n]



count = 0


def to_simplified_sentence_list(x):
    global count, stop_words
    count += 1
    if count % 10000 == 0:
        print(count)
    
    # Lower case
    x = x.lower()

    # Only keep alphabet, space and apostrophe characters
    x = re.sub(r"[^a-z '-]+", "", x)
    x = re.sub(r" [-']", " ", x)
    x = re.sub(r"[-'] ", " ", x)
    x = re.sub(r" +", " ", x)
    x = x.strip()

    # Convert to array
    if len(x) > 0:
        x = x.split(" ")
        # remove stop words
        x = [e for e in x if not e in stop_words]
    else:
        x = []
    
    return x


def add_dev_n_pub(x):
    count += 1
    if count % 10000 == 0:
        print(count)
    
    result = x["pre_word_embedding"]
    result.append(x["type"])

    dev_list = [studio_dict[str(e)] for e in x["developers"]]
    result += [studio_dict[str(e)] for e in x["publishers"]]
    return result


def add_genres(x):
    global count, lang_key, lang_name
    
    count += 1
    if count % 10000 == 0:
        print(count)

    result = x["pre_word_embedding"].copy()
    result += [genre_dict[str(genre)] for genre in x["genres"]]
    return result


lang_key = ["lang_en", "lang_fr", "lang_ge", "lang_es", "lang_po",
            "lang_zh", "lang_ja", "lang_ko", "lang_it", "lang_ru", "lang_ar"]
lang_name = ["english", "french", "german", "spanish", "portuguese",
             "chinese", "japanese", "korean", "italian", "russian", "arabic"]
def add_language(x):
    global count, lang_key, lang_name
    
    count += 1
    if count % 10000 == 0:
        print(count)

    result = x["pre_word_embedding"].copy()
    result += [lang_name[i] for i in range(len(lang_key)) if x[lang_key[i]]]
    return result


def get_average_vector(x, model):
    vector_list = []
    for e in x:
        try:   # Some words only appear once
            vector_list.append(model.wv[e])
        except:
            continue

    vector_list = np.asarray(vector_list)
    result = vector_list.mean(axis=0)
    return result

