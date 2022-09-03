from os import getcwd, path

def init_strings():
    s = ""
    dummy = 1
    s += f"from evo_playground.parameters.parameters{dummy:03d} import Parameters as p{dummy:03d}\n"

    for i in range(100, 107):
        s += f"from evo_playground.parameters.parameters{i:02d} import Parameters as p{i}\n"

    s += '\nTEST = [p001]\n\n'
    sm_batch = 'SM_BATCH_00 = ['
    for smn in range(100, 107):
        sm_batch += f'p{smn}, '
    sm_batch += ']\n'
    s += sm_batch
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
