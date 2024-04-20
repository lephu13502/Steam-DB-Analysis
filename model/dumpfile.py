# count = 0

# def html_string_to_json(x):
#     try:
#         if pd.isna(x):
#             return {}
#         else:
#             return json.loads(x.replace("\"", "").replace("\'", "\""))
#     except:
#         raise Exception(x)

# def get_minimum_requirement(x):
#     global count
#     if count % 10000 == 0:
#         print(f"Count at {count}")
#     count += 1
#     try:
#         temp = json.loads(x.replace("\"", "").replace("\'", "\""))
#         bs_model = BeautifulSoup(temp["minimum"])
#         return bs_model.text.replace("Minimum:", "")
#     except:
#         return None

# def get_recommend_requirement(x):
#     global count
#     if count % 10000 == 0:
#         print(f"Count at {count}")
#     count += 1
#     try:
#         temp = json.loads(x.replace("\"", "").replace("\'", "\""))
#         bs_model = BeautifulSoup(temp["recommended"])
#         return bs_model.text.replace("Recommended:", "")
#     except:
#         return None




# count = 0
# df["min_require"] = df["pc_requirements"].apply(get_minimum_requirement)
# print()
# count = 0
# df["rec_require"] = df["pc_requirements"].apply(get_recommend_requirement)
# df.drop(columns=["pc_requirements"], inplace=True)





# def get_average_age_rating(x, method=0):

#     avg_val = 0
#     val_list = []
    
#     try:
#         if not isinstance(x, str):
#             return x
            
#         json_x = json.loads(x.replace("\'", "\""))
#         if "dejus" in json_x:
#             keyword = "rating" if "rating" in json_x["dejus"] else "required_age"
#             try:
#                 val_list.append(int(json_x["dejus"][keyword].replace("a", "").replace("l", "0")))
#             except:
#                 pass
    
#         if "cero" in json_x:
#             keyword = "rating" if "rating" in json_x["cero"] else "required_age"
#             try:
#                 match json_x["cero"][keyword]:
#                     case "a": val_list.append(0)
#                     case "b": val_list.append(12)
#                     case "c": val_list.append(15)
#                     case "d": val_list.append(17)
#                     case "z": val_list.append(18)
#             except:
#                 pass
    
#         if "esrb" in json_x:
#             keyword = "rating" if "rating" in json_x["esrb"] else "required_age"
#             try:
#                 match json_x["esrb"][keyword]:
#                     case "c": val_list.append(3)
#                     case "e": val_list.append(7)
#                     case "e10": val_list.append(12)
#                     case "t": val_list.append(16)
#                     case "m": val_list.append(17)
#                     case "a": val_list.append(18)
#             except:
#                 pass
    
#         if "pegi" in json_x:
#             keyword = "rating" if "rating" in json_x["pegi"] else "required_age"
#             try:
#                 val_list.append(int(json_x["pegi"][keyword]))
#             except:
#                 pass
    
#         if "usk" in json_x:
#             keyword = "rating" if "rating" in json_x["usk"] else "required_age"
#             try:
#                 val_list.append(int(json_x["usk"][keyword]))
#             except:
#                 pass
    
#         if len(val_list) == 0:
#             return 0
#         else:
#             for val in val_list:
#                 avg_val += val
#             avg_val /= len(val_list)
#             return avg_val

#     except Exception as e:
#         match method:
#             case 0:
#                 return get_average_age_rating(x.replace("You\'re", "U re")  \
#                                                .replace("\'CI\'", "") \
#                                                .replace(" d\'", " d")  \
#                                                .replace("t\'s", "t is"), 1)
#             case 1:
#                 return None




# df["avg_age"] = df["ratings"].apply(get_average_age_rating)
# df.drop(columns=["ratings"], inplace=True)