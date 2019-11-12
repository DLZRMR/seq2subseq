import sys
sys.path.append('preprocess')
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import refit_cfg
import os
import random
from sklearn.model_selection import train_test_split


name = ['WashingMachine', 'Kettle', 'Microwave', 'Fridge', 'Dishwasher']
appliance_dict = {
    'WashingMachine': refit_cfg.washingmachine,
    'Kettle': refit_cfg.kettle,
    'Microwave': refit_cfg.microwave,
    'Fridge': refit_cfg.fridge,
    'Dishwasher': refit_cfg.dishwasher
}


def align_process(house_id):
    data = np.load('data\\REFIT\\original_data\\%d.npy' % house_id)
    new_data = []
    current_index = 0
    current_time = int(data[0][0])
    end_time = int(data[-1][0]) + 8
    interval_threshold = refit_cfg.separation_threshold
    isend = 0
    data_length = len(data)

    while current_time <= end_time:
        current_interval = int(data[current_index+1][0]) - int(data[current_index][0])
        if current_interval < interval_threshold:   # small interval
            if current_time > int(data[current_index][0]):
                temp_index = current_index + 1
                while current_time > int(data[temp_index][0]):
                    temp_index += 1
                    if temp_index > (data_length-1):
                        temp_index -= 1
                        break

                if abs(current_time - int(data[temp_index-1][0])) > abs(int(data[temp_index][0])-current_time):
                    current_index = temp_index
                    if temp_index == (data_length-1):
                        print('The end!')
                        isend = 1
                else:
                    current_index = temp_index - 1
            t = []
            for element in data[current_index]:
                t.append(element)
            t[0] = current_time
            new_data.append(t)
            if isend == 1:
                break
            current_time += 8
            if current_index % 1000 == 0:
                print('House %d processing: %f' % (house_id, current_index/data_length))
        else:   # big interval
            current_index += 1
            current_time = int(data[current_index][0])

    np.save('data\\REFIT\\after_align\\%d.npy' % house_id, new_data)


def visual(house_id, channel_id, start, length):
    data = np.load('data\\REFIT\\after_align\\%d.npy' % house_id)
    print(len(data))
    target = []
    c = channel_id+1
    for r in data:
        target.append(int(r[c]))
    y = target[start:start+length]
    plt.plot(y)
    plt.show()


def diff(house_id):
    data = np.load('data\\REFIT\\after_align\\%d.npy' % house_id)
    d = []
    for i in range(len(data)-1):
        d.append(int(data[i+1][0])-int(data[i][0]))
    plt.plot(d)
    plt.show()
    plt.close()


def appliance_separation(dict, appliance_name):
    path = 'data\\REFIT\\appliance_data\\%s' % appliance_name
    if not os.path.exists(path):
        os.mkdir(path)

    for house_id, channel_id in dict.items():
        data = np.load('data\\REFIT\\after_align\\%d.npy' % house_id)
        appliance_data = []
        for row in data:
            appliance_data.append([row[1], row[channel_id+1]])
        np.save(os.path.join(path, '%d_%d.npy' % (house_id, channel_id)), appliance_data)
        print('Appliance %s  House %d complete!' % (appliance_name, house_id))


def show_appliance(house_id, appliance_name):
    channel_id = appliance_dict[appliance_name][house_id]
    data = np.load('data\\REFIT\\after_align\\%s\\%d_%d.npy' % (appliance_name, house_id, channel_id))
    print(len(data))
    mains = []
    app = []
    for i in data:
        mains.append(int(i[0]))
        app.append(int(i[1]))
    plt.figure(figsize=(20, 8))
    plt.plot(mains)
    plt.plot(app)
    plt.show()


def cull(cull_dict):
    for appliance_name, _dict in cull_dict.items():
        path = 'data\\REFIT\\after_culling\\%s' % appliance_name
        if not os.path.exists(path):
            os.mkdir(path)
        for house_id, cull_list in _dict.items():
            channel_id = appliance_dict[appliance_name][house_id]
            data = np.load('data\\REFIT\\after_align\\%s\\%d_%d.npy' % (appliance_name, house_id, channel_id))
            new_data = []
            _cull_list = [[0, cull_list[0][0]]]
            for i in range(len(cull_list)-1):
                _cull_list.append([cull_list[i][1], cull_list[i+1][0]])
            _cull_list.append([cull_list[-1][1], (len(data)-1)])

            for i in _cull_list:
                if i[1] - i[0] != 0:
                    for j in range(i[0], i[1]):
                        new_data.append(data[j])
            np.save('data\\REFIT\\after_culling\\%s\\%d_%d.npy' % (appliance_name, house_id, channel_id), new_data)
            print('House %d  %s  complete!' % (house_id, appliance_name))


def appliance_separation(dict, appliance_name):
    """
    将各个电器的数据进行分解，放置到appliance_data文件夹下对应电器的文件夹中，以house_id和channel_id进行命名
    :param dict: 电器数据来源
    :param appliance_name: 当前电器的名称，用以创建文件夹
    :return:
    """
    path = 'data\\REFIT\\appliance_data\\%s' % appliance_name
    if not os.path.exists(path):
        os.mkdir(path)

    for house_id, channel_id in dict.items():
        data = np.load('data\\REFIT\\after_align\\%d.npy' % house_id)
        appliance_data = []
        for row in data:
            appliance_data.append([row[1], row[channel_id+1]])      # 将mains 和 appliance 作为一条单独的记录
        np.save(os.path.join(path, '%d_%d.npy' % (house_id, channel_id)), appliance_data)
        print('Appliance %s  House %d complete!' % (appliance_name, house_id))


def show_appliance(house_id, appliance_name):
    """
    具体观察每个电器的图形表示，将大段的数据缺失或者数据错误进行标注，构造cull_dict字典，在cull进行片段删除
    :param house_id:
    :param appliance_name:
    :return:
    """
    channel_id = appliance_dict[appliance_name][house_id]
    data = np.load('data\\REFIT\\after_culling\\%s\\%d_%d.npy' % (appliance_name, house_id, channel_id))
    print(len(data))
    mains = []
    app = []
    for i in data:
        mains.append(int(i[0]))
        app.append(int(i[1]))
    plt.figure(figsize=(20, 8))
    plt.plot(mains)
    plt.plot(app)
    plt.show()


def cull(cull_dict):
    """
    根据画的图，将大段的空缺段进行删除，删除之后，需要进行比对
    :param cull_dict:
    :return:
    """
    for appliance_name, _dict in cull_dict.items():
        path = 'data\\REFIT\\after_culling_2\\%s' % appliance_name
        if not os.path.exists(path):
            os.mkdir(path)
        for house_id, cull_list in _dict.items():
            channel_id = appliance_dict[appliance_name][house_id]
            data = np.load('data\\REFIT\\after_culling_2\\%s\\%d_%d.npy' % (appliance_name, house_id, channel_id))
            new_data = []
            # 对cull_list进行变形，变成表征合理数据的区间
            _cull_list = [[0, cull_list[0][0]]]
            for i in range(len(cull_list)-1):
                _cull_list.append([cull_list[i][1], cull_list[i+1][0]])
            _cull_list.append([cull_list[-1][1], (len(data)-1)])

            for i in _cull_list:
                if i[1] - i[0] != 0:
                    for j in range(i[0], i[1]):
                        new_data.append(data[j])
            np.save('data\\REFIT\\after_culling_2\\%s\\%d_%d.npy' % (appliance_name, house_id, channel_id), new_data)
            print('House %d  %s  complete!' % (house_id, appliance_name))


def separate(appliance_name):
    window_width = refit_cfg.window_width[appliance_name]
    data_path = 'data\\REFIT\\after_culling\\%s' % appliance_name
    count = 0
    appliance_train_validation = []
    appliance_test = []
    main_train_validation = []
    main_test = []

    for house_id, channel_id in refit_cfg.train_validation[appliance_name].items():
        # train & validation
        appliance_train_validation.clear()
        main_train_validation.clear()
        data = np.load(os.path.join(data_path, '%s_%s.npy' % (house_id, channel_id)))
        current_head = 0
        data_length = len(data)
        end = data_length - window_width - 1
        while current_head < end:
            temp_main = []
            temp_appliance = []
            for i in range(current_head, current_head+window_width):
                temp_main.append(data[i][0])
                temp_appliance.append(data[i][1])
            r = random.random()
            current_head += int(window_width*r)
            appliance_train_validation.append(temp_appliance)
            main_train_validation.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('T & V 1： House %d   %f' % (house_id, (current_head / data_length)))

        data_length -= window_width
        random_clip = refit_cfg.random_clip[appliance_name]
        for i in range(random_clip):
            r = random.random()
            start = int(r*data_length)
            temp_main = []
            temp_appliance = []
            for j in range(start, start + window_width):
                temp_main.append(data[j][0])
                temp_appliance.append(data[j][1])
            appliance_train_validation.append(temp_appliance)
            main_train_validation.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('T & V 2： House %d   %f' % (house_id, (i / random_clip)))
        print('Train & Validation: House %d %s complete!' % (house_id, appliance_name))
        np.save(os.path.join(data_path, '1024\\appliance_train_validation_%d.npy' % house_id), appliance_train_validation)
        np.save(os.path.join(data_path, '1024\\main_train_validation_%d.npy' % house_id), main_train_validation)


    # test
    count = 0
    for house_id, channel_id in refit_cfg.test[appliance_name].items():
        appliance_test.clear()
        main_test.clear()
        data = np.load(os.path.join(data_path, '%s_%s.npy' % (house_id, channel_id)))
        current_head = 0
        data_length = len(data)
        end = data_length - window_width - 1
        while current_head < end:
            temp_main = []
            temp_appliance = []
            for i in range(current_head, current_head+window_width):
                temp_main.append(data[i][0])
                temp_appliance.append(data[i][1])
            r = random.random()
            current_head += int(r*window_width)
            appliance_test.append(temp_appliance)
            main_test.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('Test 1： House %d   %f' % (house_id, (current_head / data_length)))

        data_length -= window_width
        for i in range(refit_cfg.random_clip[appliance_name]):
            r = random.random()
            start = int(r*data_length)
            temp_main = []
            temp_appliance = []
            for j in range(start, start + window_width):
                temp_main.append(data[j][0])
                temp_appliance.append(data[j][1])
            appliance_test.append(temp_appliance)
            main_test.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('Test 2： House %d   %f' % (house_id, (i / data_length)))
        print('Test 2: House %d %s complete!' % (house_id, appliance_name))
        np.save(os.path.join(data_path, '1024\\appliance_test_%d.npy' % house_id), appliance_test)
        np.save(os.path.join(data_path, '1024\\main_test_%d.npy' % house_id), main_test)


def clip_visual(appliance_name):
    base_path = 'data\\REFIT\\after_culling\\%s' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_train_.npy'))
    main_data = np.load(os.path.join(base_path, 'main_train_.npy'))
    print('Data load complete!')
    loop = 1000
    x = np.linspace(256, 768, 512)
    length = len(appliance_data)
    for i in range(loop):
        r = int(random.random()*length)
        plt.figure(figsize=(25, 10), dpi=100)
        plt.subplot(211)
        plt.xlim(0, 1024)
        plt.plot(main_data[r])
        plt.subplot(212)
        plt.xlim(0, 1024)
        plt.plot(x, appliance_data[r])
        savefig(os.path.join(base_path, 'clip_view\\%d.jpg' % i))
        plt.close()


def train_validation_split(appliance_name):
    data_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance = np.load(os.path.join(data_path, 'appliance_train_validation.npy'))
    main = np.load(os.path.join(data_path, 'main_train_validation.npy'))
    appliance_train, appliance_validation, main_train, main_validation = \
        train_test_split(appliance, main, test_size=0.2)
    print(len(appliance_train))
    print(len(main_train))

    np.save(os.path.join(data_path, 'appliance_train.npy'), appliance_train)
    np.save(os.path.join(data_path, 'main_train.npy'), main_train)
    np.save(os.path.join(data_path, 'appliance_validation.npy'), appliance_validation)
    np.save(os.path.join(data_path, 'main_validation.npy'), main_validation)


def data_integration(appliance_name):
    data_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance = []
    main = []
    for house_id, channel_id in refit_cfg.train_validation[appliance_name].items():
        appliance_data = np.load(os.path.join(data_path, 'appliance_train_validation_%d.npy' % house_id))
        main_data = np.load(os.path.join(data_path, 'main_train_validation_%d.npy' % house_id))
        for i in appliance_data:
            appliance.append(i)
        for i in main_data:
            main.append(i)

    print(len(appliance))
    print(len(main))
    np.save(os.path.join(data_path, 'appliance_train_validation.npy'), appliance)
    np.save(os.path.join(data_path, 'main_train_validation.npy'), main)

    appliance_test = []
    main_test = []
    for house_id, channel_id in refit_cfg.test[appliance_name].items():
        appliance_data = np.load(os.path.join(data_path, 'appliance_test_%d.npy' % house_id))
        main_data = np.load(os.path.join(data_path, 'main_test_%d.npy' % house_id))
        for i in appliance_data:
            appliance_test.append(i)
        for i in main_data:
            main_test.append(i)

    print(len(appliance_test))
    print(len(main_test))
    np.save(os.path.join(data_path, 'appliance_test.npy'), appliance_test)
    np.save(os.path.join(data_path, 'main_test.npy'), main_test)


def positive_negative(appliance_name):
    base_path = 'data\\REFIT\\after_culling\\%s' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_train.npy'))
    count = 0
    threshold = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    d = {}
    for i in range(len(threshold)):
        d[threshold[i]] = 0
    print(d)

    for th in threshold:
        for i in appliance_data:
            sum = 0
            for j in i:
                sum += int(j)
            if sum > th:
                d[th] += 1
        print('Thres %d complete!' % th)

    for thres, count in d.items():
        print('Thres: %d   %d/%d   %f' % (thres, count, len(appliance_data), count/len(appliance_data)))


def clip_view(appliance_name, thres):
    base_path = 'data\\REFIT\\after_culling\\%s' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_train.npy'))
    count = 0

    for i in appliance_data:
        sum = 0
        for j in i:
            sum += int(j)
        if sum > thres:
            plt.figure(figsize=(25, 10), dpi=100)
            plt.plot(i.astype(int))
            savefig(os.path.join(base_path, 'clip_view\\%d.jpg' % count))
            plt.close()
            count += 1


def test_process(appliance_name):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_test_512.npy'))
    temp = [0.0]*512
    new_app = []
    for i in range(len(appliance_data)):
        max = np.max(appliance_data[i])
        if max < 0.05:
            print(max)
            new_app.append(temp)
        else:
            new_app.append(appliance_data[i])
    np.save(os.path.join(base_path, 'appliance_test_512.npy'), new_app)


def separate_positive_negative(appliance_name, thres, peak):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_train.npy'))
    main_data = np.load(os.path.join(base_path, 'main_train.npy'))
    count = 0
    appliance_positive = []
    appliance_negative = []
    main_positive = []
    main_negative = []
    appliance_temp = [0] * 1024

    for i in range(len(appliance_data)):
        sum = 0
        max = 0
        for j in appliance_data[i]:
            sum += int(j)
        for j in range(512):
            if int(appliance_data[i][j+256]) > max:
                max = int(appliance_data[i][j+256])
        if max < peak:
            sum = 0
        if sum > thres:
            appliance_positive.append(appliance_data[i])
            main_positive.append(main_data[i])
        else:
            appliance_negative.append(appliance_temp)
            main_negative.append(main_data[i])
        if i % 1000 == 0:
            print('Processing: %f' % (i/len(appliance_data)))

    np.save(os.path.join(base_path, 'appliance_positive.npy'), appliance_positive)
    np.save(os.path.join(base_path, 'main_positive.npy'), main_positive)
    np.save(os.path.join(base_path, 'appliance_negative.npy'), appliance_negative)
    np.save(os.path.join(base_path, 'main_negative.npy'), main_negative)


def generate_balanced_dataset(appliance_name, negative_ratio):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_positive = list(np.load(os.path.join(base_path, 'appliance_positive.npy')))
    appliance_negative = np.load(os.path.join(base_path, 'appliance_negative.npy'))
    main_positive = list(np.load(os.path.join(base_path, 'main_positive.npy')))
    main_negative = np.load(os.path.join(base_path, 'main_negative.npy'))
    print('Data load complete!')

    positive_length = len(appliance_positive)
    negative_length = len(appliance_negative)
    print('Postive length: %d   negative length: %d' % (positive_length, negative_length))
    for i in range(int(positive_length*negative_ratio)):
        r = int(random.random()*negative_length)
        appliance_positive.append(appliance_negative[r])
        main_positive.append(main_negative[r])
    print('Data generate complete! length: %d' % (len(appliance_positive)))

    index = np.linspace(0, len(appliance_positive)-1, len(appliance_positive)).astype(int)
    random.shuffle(index)
    appliance_new = []
    main_new = []

    for i in index:
        appliance_new.append(appliance_positive[i])
        main_new.append(main_positive[i])
    print('Data shuffle complete!')

    np.save(os.path.join(base_path, 'appliance_train_balanced.npy'), appliance_new)
    np.save(os.path.join(base_path, 'main_train_balanced.npy'), main_new)
    print('Data save complete!')


def shrink(appliance_name, scale):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_train_balanced.npy'))
    main_data = np.load(os.path.join(base_path, 'main_train_balanced.npy'))
    appliance_new = []
    main_new = []
    print('Data load complete!')

    for i in range(len(appliance_data)):
        appliance_temp = []
        main_temp = []
        for j in range(len(appliance_data[i])):
            appliance_temp.append(float(int(appliance_data[i][j])/scale))
        for j in range(len(main_data[i])):
            main_temp.append(float(int(main_data[i][j])/scale))
        appliance_new.append(appliance_temp)
        main_new.append(main_temp)
    print('Process complete!')

    np.save(os.path.join(base_path, 'appliance_train_%d.npy' % scale), appliance_new)
    np.save(os.path.join(base_path, 'main_train_%d.npy' % scale), main_new)


def shrink_validation(appliance_name, scale):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_validation.npy'))
    main_data = np.load(os.path.join(base_path, 'main_validation.npy'))
    appliance_new = []
    main_new = []
    print('Data load complete!')

    for i in range(len(appliance_data)):
        appliance_temp = []
        main_temp = []
        for j in range(len(appliance_data[i])):
            appliance_temp.append(float(int(appliance_data[i][j])/scale))
        for j in range(len(main_data[i])):
            main_temp.append(float(int(main_data[i][j])/scale))
        appliance_new.append(appliance_temp)
        main_new.append(main_temp)
    print('Process complete!')

    np.save(os.path.join(base_path, 'appliance_validation_%d.npy' % scale), appliance_new)
    np.save(os.path.join(base_path, 'main_validation_%d.npy' % scale), main_new)


def appliance_1024to512(appliance_name):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_train = np.load(os.path.join(base_path, 'appliance_train_1000.npy'))
    appliance_validation = np.load(os.path.join(base_path, 'appliance_validation_1000.npy'))
    appliance_test = np.load(os.path.join(base_path, 'appliance_test_1000.npy'))
    at_new = []
    av_new = []
    ae_new = []

    for i in range(len(appliance_train)):
        at_temp = []
        for j in range(256, 768):
            at_temp.append(float(appliance_train[i][j]))
        at_new.append(at_temp)
    for i in range(len(appliance_validation)):
        av_temp = []
        for j in range(256, 768):
            av_temp.append(float(appliance_validation[i][j]))
        av_new.append(av_temp)
    for i in range(len(appliance_test)):
        ae_temp = []
        for j in range(256, 768):
            ae_temp.append(float(appliance_test[i][j]))
        ae_new.append(ae_temp)

    np.save(os.path.join(base_path, 'appliance_train_512.npy'), at_new)
    np.save(os.path.join(base_path, 'appliance_validation_512.npy'), av_new)
    np.save(os.path.join(base_path, 'appliance_test_512.npy'), ae_new)


def shrink_test(appliance_name, scale):
    base_path = 'data\\REFIT\\after_culling\\%s\\1024' % appliance_name
    appliance_data = np.load(os.path.join(base_path, 'appliance_test.npy'))
    main_data = np.load(os.path.join(base_path, 'main_test.npy'))
    appliance_new = []
    main_new = []
    print('Data load complete!')

    for i in range(len(appliance_data)):
        appliance_temp = []
        main_temp = []
        for j in range(len(appliance_data[i])):
            appliance_temp.append(float(int(appliance_data[i][j])/scale))
        for j in range(len(main_data[i])):
            main_temp.append(float(int(main_data[i][j])/scale))
        appliance_new.append(appliance_temp)
        main_new.append(main_temp)
    print('Process complete!')

    np.save(os.path.join(base_path, 'appliance_test_1000.npy'), appliance_new)
    np.save(os.path.join(base_path, 'main_test_1000.npy'), main_new)


if __name__ == '__main__':
    appliance_name = 'WashingMachine'
    separate(appliance_name)
    data_integration(appliance_name)
    train_validation_split(appliance_name)
    separate_positive_negative(appliance_name, 1500, 20)
    generate_balanced_dataset(appliance_name, 1)
    shrink(appliance_name, 1000)
    shrink_validation(appliance_name, 1000)
    shrink_test(appliance_name, 1000)
    appliance_1024to512(appliance_name)
    # test_process(appliance_name)
    print('Process complete!!!')
