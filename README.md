A Pythonic translation of [ridgeModel](https://github.com/musall/ridgeModel) in development, meant to process data created with the [wfield](https://github.com/jcouto/wfield) package. To run:

1) Install conda and run ``conda env create -f env.yml`` inside the package folder.
2) Run ``conda activate ridgemodel`` and install ridgemodel using the command ``python setup.py install``
3) Type ``ridgemodel -h`` to see available commands.

For an example usage, place the outputs of the wfield package (U.npy/U_atlas.npy and SVTcorr.npy) into the same folder as the files in ``for_wfield_outputs``, and run ``ridgemodel process`` followed by the folder name.