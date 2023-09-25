import pickle

eval_res_dict = pickle.load(open('./10', "rb"))
#print(eval_res_dict)

print("=============elo===============")
print(eval_res_dict['elo'])

print("=============update_type is static===============")
print(eval_res_dict['update_type is static'])

print("=============reward_hyperparams===============")
for item in eval_res_dict['reward_hyperparams']:
    print(item)

print("=============battle_result===============")
for item in eval_res_dict['battle_result']:
    print(item)
