from os import getcwd, path

def init_strings():
    s = ""
    dummy = 1
    s += f"from evo_playground.parameters.parameters{dummy:03d} import Parameters as p{dummy:03d}\n"

    for i in range(100, 107):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for i in range(110, 117):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for i in range(120, 127):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for i in range(130, 172):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for i in range(230, 272):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for i in range(300, 327):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    for i in range(400, 416):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"
    s += '\nTEST = [p001]\n\n'
    sm_batch0 = 'SM_BATCH_00 = ['
    for smn in range(100, 107):
        sm_batch0 += f'p{smn}, '
    sm_batch0 += ']\n'
    s += sm_batch0

    sm_batch1 = 'SM_BATCH_01 = ['
    for smn in range(110, 117):
        sm_batch1 += f'p{smn}, '
    sm_batch1 += ']\n'
    s += sm_batch1

    sm_batch2 = 'SM_BATCH_02 = ['
    for smn in range(120, 127):
        sm_batch2 += f'p{smn}, '
    sm_batch2 += ']\n'
    s += sm_batch2

    big_batch_00 = 'BIG_BATCH_00 = ['
    for smn in range(130, 172):
        big_batch_00 += f'p{smn}, '
    big_batch_00 += ']\n'
    s += big_batch_00

    test_01 = 'TEST_01 = ['
    for smn in range(230, 272):
        test_01 += f'p{smn}, '
    test_01 += ']\n'
    s += test_01

    big_batch_01 = 'BIG_BATCH_01 = ['
    for smn in range(300, 324):
        big_batch_01 += f'p{smn}, '
    big_batch_01 += ']\n'
    s += big_batch_01

    big_batch_02 = 'BIG_BATCH_02 = ['
    for smn in range(400, 416):
        big_batch_02 += f'p{smn}, '
    big_batch_02 += ']\n'
    s += big_batch_02

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
