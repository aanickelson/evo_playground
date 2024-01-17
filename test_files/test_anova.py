import pandas as pd
import pingouin as pg
from evo_playground.test_files.test_anova_data import data as dt
import csv


def pd_from_data(metafile, data):
    metadf = pd.read_csv(metafile)
    dom = 'Hopper'

    items = []
    for [bh_nm, d] in data:
        metadata = metadf[(metadf['behavior'] == bh_nm) & (metadf['domain'] == dom)].values.flatten().tolist()
        metadata.append(d)
        items.append(metadata)

    col_names = metadf.columns.tolist()
    col_names.append('value')
    df = pd.DataFrame(items, columns=col_names)
    return df


if __name__ == '__main__':
    metafn = '/home/anna/Downloads/Behavior defs metadata.csv'
    print(pd_from_data(metafn, dt))

# https://ethanweed.github.io/pythonbook/05.05-anova2.html#factorial-anova-versus-one-way-anovas
# print(pd.DataFrame(round(df.groupby(by=['Raw / summary', 'Policy output'])['100000 mean'].mean(),2)).reset_index())
# model1 = pg.anova(dv='100000 mean', between='Policy output', data=df, detailed=True)
# round(model1, 2)
# print(model1)


