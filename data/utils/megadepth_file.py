l = ['0000', '0001', '0002', '0003', '0004', '0005', '0007', '0008', '0011', '0012', '0013', '0015', '0016',\
     '0017', '0019', '0020', '0021', '0023', '0024', '0025', '0026', '0027', '0032', '0033', '0034', '0035',\
     '0036', '0205', '0212', '0214', '0217', '0223', '0224', '0229', '0231', '0235', '0237', '0238', '0240',\
     '0243', '0252', '0257', '0258', '0265', '0269', '0271', '0277', '0281', '0285', '0286', '0430', '0443',\
     '0446', '0455', '0472', '0474', '0478', '0482', '0493', '0496', '0505', '0559', '0733', '0768', '0860',\
     '1001', '1017', '1589', '3346', '4541', '5000', '5001', '5002', '5003', '5004', '5005', '5006', '5008',\
     '5009', '5010', '5011', '5012', '5013', '5014', '5015', '5016', '5017', '5018', '0099', '0100', '0101',\
     '0102', '0103', '0104', '0105', '0107', '0115', '0117', '0121', '0122', '0130', '0133', '0137', '0141',\
     '0143', '0147', '0148', '0149', '0150', '0151', '0156', '0160', '0162', '0168', '0175', '0176', '0177',\
     '0178', '0183', '0185', '0186', '0189', '0190', '0197', '0200', '0039', '0041', '0042', '0043', '0044',\
     '0046', '0047', '0048', '0050', '0056', '0057', '0058', '0060', '0061', '0062', '0063', '0064', '0065',\
     '0067', '0070', '0071', '0076', '0078', '0080', '0083', '0086', '0087', '0090', '0094', '0095', '0037',\
     '0098', '0204', '0290', '0412', '0294', '0306', '0307', '0312', '0323', '0326', '0327', '0331', '0335',\
     '0341', '0348', '0349', '0360', '0366', '0377', '0380', '0387', '0389', '0394', '0402', '0406', '0407', '0411']

# import pandas as pd
# a = '/home/user/computer_vision/pairs.txt'
# df = pd.read_csv(a, sep=' ', header=None)
# print(df.iloc[0])

def read_dataset_pairs(txt_path='/home/user/computer_vision/pairs.txt', max_scale=10.):
    '''
    Function read the dataset pairs
        :param txt_path: path to the dataset pairs
        :param max_scale: optional parameter to control the maximum scale value
        :return: dataset pairs
    '''

    file1 = open(txt_path, 'r')
    pairs = file1.readlines()

    scale_dataset = []
    for idx in range(len(pairs)):
        info_pair = pairs[idx].split('-/-')
        scale_ratio = float(info_pair[7].split('\n')[0])

        if scale_ratio < max_scale and scale_ratio > 1./max_scale:
            new_sample = {'img1': info_pair[5], 'img2': info_pair[6],
                          'scene': info_pair[0], 'scale_ratio': scale_ratio}
            scale_dataset.append(new_sample)

    return scale_dataset


file1 = open('/home/user/computer_vision/pairs.txt', 'r')
pairs = file1.readlines()
a = pairs[0]
info_pairs = a.split(' ')
print(len(info_pairs))