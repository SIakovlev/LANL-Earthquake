import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
from dfc import MemoryManager
import os

from utils import str_to_class

if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    data_fname = kwargs['data_fname']
    data_fname_dest = kwargs['data_fname_dest']

    # 1. Parse params and create a chain of processing instances
    processors = []
    for p in kwargs['processing']:
        class_ = str_to_class('DataProcessing', p['name'])
        processor = class_(**p)
        processors.append(processor)

    # 2. Load data
    # TODO: implement "smart" data loading to handle data too big to fit in the memory

    df_handler = MemoryManager(chunk_size=1e6)
    df_handler.set_hdf_iterator(data_fname)
    # df_handler = processors[0].load(data_fname, chunk_size=1e6)

    # for df in df_handler:
    #     # 3. Run processing
    #     print('.......................Processing started.........................')
    #     for i, p in enumerate(processors):
    #         print(f'{i}: name={p.__class__.__name__} | columns={p.column_names}')
    #         df = p(df)
    #
    #     # 4. Save modified dataframe
    #     # processors[0].save(df, data_fname_dest)
    #     df_handler.save(df, data_fname_dest, chunk_size_MB=3, dir_name='test_dataset')
    #     print(f'dataframe saved as {data_fname_dest}')
    # pd.set_option('display.max_columns', 500)
    # print('.......................Processing finished.........................')
    # print(df.head(10))
    df_handler.set_iterator(os.path.join(data_fname_dest, 'test_dataset/'))
    df_handler.check_integrity()
    # print(os.path.join(data_fname_dest, 'test_dataset/'))
    print("500th element: {}".format(df_handler.iloc[500]))
    print("5000th element: {}".format(df_handler.iloc[5000]))
    print("50000th element: {}".format(df_handler.iloc[50000]))
    print("500000th element: {}".format(df_handler.iloc[500000]))
    print()
    print("500 + 502 elements: {}".format(df_handler.iloc[[500, 502]]))
    print("50000 + 500000 element: {}".format(df_handler.iloc[[50000, 500000]]))
    print()
    print(df_handler.iloc[1:70000].tail(10))
    print()
    # print(df_handler['s'].head(10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str)
    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)
