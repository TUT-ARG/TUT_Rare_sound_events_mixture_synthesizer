__author__      = "Aleksandr Diment"
__email__       = "aleksandr.diment@tut.fi"

import argparse, textwrap, os
from core import main
from IPython import embed

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Rare event detection
            Mixture synthesizer
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Aleksandr Diment ( aleksandr.diment@tut.fi )

            This software performs generation of mixture recipes and synthesis of the mixtures for the DCASE 2017
            Rare sound event detection task. Each mixture consists of at most one rare sound event of the target class.
            The generation uses data provided by the DCASE2017 challenge consisting of backround recordings and target
            isolated sound events. For more details, see http://www.cs.tut.fi/sgn/arg/dcase2017/


        '''))

    parser.add_argument("-data_path", help="Path to data (should include at least source_data "
                                           "folder with forlders bgs, events, cv_setup). "
                                           "Default: {}".format(os.path.join('..', 'data')),
                        default=os.path.join('..', 'data'), dest='data_path')
    parser.add_argument('-params', help='Parameter file (yaml) for the devtrain mixtures '
                                                 '(default: the provided mixing_params.yaml). ',
                        required=False, default='mixing_params.yaml', dest='params')
    args = parser.parse_args()

    main(data_path=args.data_path,
         generate_devtrain_recipes=True,
         generate_devtest_recipes=False,
         synthesize_devtrain_mixtures=True,
         synthesize_devtest_mixtures=False,
         devtrain_mixing_param_file=args.params)