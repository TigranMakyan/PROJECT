# from dataset import create_datasets
# from utils import create_cfg_lists

# cfg_path = './first40.txt'
# train_paths, val_paths, test_paths = create_cfg_lists(cfg_path)
# #CREATE DATASETS FOR TRAINING
# train_dataset, _, _ = create_datasets(
#     train_paths, val_paths, test_paths, '/media/storagedrive/Megadepth/', None, None
# )

# l_01 = []                                               # < 0.1
# l_01_02 = []                                            # 0.1 - 0.2
# l_02_03 = []                                            # 0.2 - 0.3
# l_03_04 = []                                            # 0.3 - 0.4
# l_04_05 = []                                            # 0.4 - 0.5
# l_05_06 = []                                            # 0.5 - 0.6
# l_06_07 = []                                            # 0.6 - 0.7
# l_07_08 = []                                            # 0.7 - 0.8
# l_08_09 = []                                            # 0.8 - 0.9
# l_09_10 = []                                            # 0.9 - 1.0
# l_10_15 = []                                            # 1.0 - 1.5
# l_15_20 = []                                            # 1.5 - 2.0
# l_20_30 = []                                            # 2.0 - 3.0
# l_30_50 = []                                            # 3.0 - 5.0
# l_50 = []                                               # 5.0 > 

# for item in train_dataset:
#     _, _, scale = item
#     scale = float(scale)
#     if scale < 0.1:
#         l_01.append(scale)
#     elif 0.1 <= scale < 0.2:
#         l_01_02.append(scale)
#     elif 0.2 <= scale < 0.3:
#         l_02_03.append(scale)
#     elif 0.3 <= scale < 0.4:
#         l_03_04.append(scale)
#     elif 0.4 <= scale < 0.5:
#         l_04_05.append(scale)
#     elif 0.5 <= scale < 0.6:
#         l_05_06.append(scale)
#     elif 0.6 <= scale < 0.7:
#         l_06_07.append(scale)
#     elif 0.7 <= scale < 0.8:
#         l_07_08.append(scale)
#     elif 0.8 <= scale < 0.9:
#         l_08_09.append(scale)
#     elif 0.9 <= scale < 1.0:
#         l_09_10.append(scale)
#     elif 1.0 <= scale < 1.5:
#         l_10_15.append(scale)
#     elif 1.5 <= scale < 2.0:
#         l_15_20.append(scale)
#     elif 2.0 <= scale < 3.0:
#         l_20_30.append(scale)
#     elif 3.0 <= scale < 5.0:
#         l_30_50.append(scale)
#     else:
#         l_50.append(scale)

# d01 = len(l_01)                                                
# d0102 = len(l_01_02)                                            
# d0203 = len(l_02_03)                                           
# d0304 = len(l_03_04)                                            
# d0405 = len(l_04_05)                                           
# d0506 = len(l_05_06)                                             
# d0607 = len(l_06_07)                                          
# d0708 = len(l_07_08)                                           
# d0809 = len(l_08_09)                                            
# d0910 = len(l_09_10)                                           
# d1015 = len(l_10_15)                                        
# d1520 = len(l_15_20)                                           
# d2030 = len(l_20_30)                                            
# d3050 = len(l_30_50)                                        
# d50 = len(l_50) 

# total = len(train_dataset)

# print('Result')
# print('0.0 - 0.1:\t', d01 / total)
# print('0.1 - 0.2:\t', d0102 / total)
# print('0.2 - 0.3:\t', d0203 / total)
# print('0.3 - 0.4:\t', d0304 / total)
# print('0.4 - 0.5:\t', d0405 / total)
# print('0.5 - 0.6:\t', d0506 / total)
# print('0.6 - 0.7:\t', d0607 / total)
# print('0.7 - 0.8:\t', d0708 / total)
# print('0.8 - 0.9:\t', d0809 / total)
# print('0.9 - 1.0:\t', d0910 / total)
# print('1.0 - 1.5:\t', d1015 / total)
# print('1.5 - 2.0:\t', d1520 / total)
# print('2.0 - 3.0:\t', d2030 / total)
# print('3.0 - 5.0:\t', d3050 / total)
# print('5.0 - 10.:\t', d50 / total)