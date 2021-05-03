# task_conf={
#     "task_itemsize":[65997, 1, 3, 10, 10]
# }

# task_conf={
#     "task_itemsize":[27892, 1019, 1019, 10, 10]
# }

task_conf={
    "task_itemsize":[645974,17880,7540, 3, 9 , 7]
}


# in case
sym_dict={}
# pretrain_itemsize=task_conf['task_itemsize'][0]

# sym_dict['GEN_Task1'] = pretrain_itemsize
# sym_dict['GEN_Task2'] = pretrain_itemsize+1
# sym_dict['GEN_Task3'] = pretrain_itemsize+2
# sym_dict['GEN_Task4'] = pretrain_itemsize+3
# sym_dict['GEN_Task5'] = pretrain_itemsize+4
# sym_dict['GEN_Task6'] = pretrain_itemsize+5
# sym_dict['GEN_Task7'] = pretrain_itemsize+6
# sym_dict['GEN_Task8'] = pretrain_itemsize+7
# sym_dict['GEN_Task9'] = pretrain_itemsize+8
# sym_dict['GEN_Task10'] = pretrain_itemsize+9
# # self.item_dict['END'] = len(self.item_dict) + 11
# sym_dict['incase_1'] = pretrain_itemsize+10
# sym_dict['incase_2'] = pretrain_itemsize+11
# sym_dict['incase_3'] = pretrain_itemsize+12
# sym_dict['incase_4'] = pretrain_itemsize+13
# sym_dict['incase_5'] = pretrain_itemsize+14
# sym_dict['incase_6'] = pretrain_itemsize+15
# sym_dict['incase_7'] = pretrain_itemsize+16
# sym_dict['incase_8'] = pretrain_itemsize+17
# sym_dict['incase_9'] = pretrain_itemsize+18
# sym_dict['incase_10'] = pretrain_itemsize+19


# embed_len=0
# for i in task_conf['task_itemsize']:
#     embed_len += i
# embed_len+=len(sym_dict)# the whole embedding size

#mask percentage for wait
maskp_task1=0.7 #large value means more weights are masked maskp_task1=0, no weights will be masked
maskp_task2=0.7
maskp_task3=0.7
maskp_task4=0.5
maskp_task5=0.5
maskp_task6=0.5
maskp_task7=0.5
maskp_task8=0.5
maskp_task9=0.5


taskID_1st=10001
taskID_2nd=10002
taskID_3rd=10003
taskID_4th=10004
taskID_5th=10005
taskID_6th=10006
taskID_7th=10007
taskID_8th=10008
taskID_9th=10009
taskID_10th=10010


