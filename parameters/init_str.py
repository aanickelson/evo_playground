from os import getcwd, path

def init_strings():
    s = ""
    for i in range(41):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for j in range(97, 460):
        s += f"from evo_playground.parameters.parameters{j:02d} import Parameters as p{j}\n"
    for k in range(500, 628):
        s += f"from evo_playground.parameters.parameters{k:02d} import Parameters as p{k}\n"

    # print(s)
    s += f"BATCH1 = [p0, p1, p2, p3, p4, p5, p6, p7, p8]\n"
    s += f"BATCH2 = [p9, p10, p11, p12, p13, p14, p15, p16, p17]\n"
    s += f"BATCH3 = [p18, p19, p20, p21, p22, p23, p24, p25, p26, p27]\n"
    s += f"BATCH3_SM = [p23, p24, p25, p26, p27]\n"
    s += f"ONLY23 = [p23]\n"
    s += f"BATCH4 = [p26]\n"
    s += f"BATCH5 = [p8, p28]\n"
    s += f"BATCH6 = [p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40]\n"
    s += f"MINI_B6 = [p30, p31]\n"
    s += f"TEST_BATCH = [p97, p98, p99]\n"

    batch_num = 10
    new_batches = []
    big_batch = 'BIG_BATCH_00 = ['
    for k in range(100, 460, 10):
        batch_s = f'BATCH_{batch_num} = ['
        tot_in_batch = 10
        # if k == 40:
        #     tot_in_batch = 8
        for i in range(tot_in_batch):
            batch_s += f'p{i + k}, '
            big_batch += f'p{i + k}, '
        big_batch += '\n'
        batch_s += ']\n'
        s += batch_s
        batch_num += 1
    big_batch += ']\n'
    s += big_batch

    big_batch = 'BIG_BATCH_01 = ['
    for k in range(500, 628, 10):
        batch_s = f'BATCH_{batch_num} = ['
        tot_in_batch = 10
        if k == 620:
            tot_in_batch = 8
        for i in range(tot_in_batch):
            batch_s += f'p{i + k}, '
            big_batch += f'p{i + k}, '
        big_batch += '\n'
        batch_s += ']\n'
        s += batch_s
        batch_num += 1
    big_batch += ']\n'
    s += big_batch
    return s


def filesave(str_to_save):
    filename = "__init__.py"
    filepath = path.join(getcwd(), filename)
    # Writing to file
    with open(filepath, "w") as fl:
        # Writing data to a file
        fl.writelines(str_to_save)


if __name__ == '__main__':
    string_to_save = init_strings()
    filesave(string_to_save)
