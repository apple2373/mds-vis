# MDS vis

## Environment 
See [`env.yml`](./env.yml) for the exact environment. As a shortcut, you can use the following python binary as is if you have access to `salk.psych.indiana.edu`. 
```
/data/stsutsui/public/mds-vis/miniconda/bin/python
```
Alternatively, you can do
```
export PATH="/data/stsutsui/public/mds-vis/miniconda/bin:$PATH"
```

Make sure that `which python` will give the above python binary.

### Installing Gist Libraries
Basically follow the instruction from https://github.com/tuttieee/lear-gist-python  and use the following for configure FFTW
```
export ROOT="/home/stsutsui/data/public/mds-vis/miniconda/"
wget http://www.fftw.org/fftw-3.3.8.tar.gz
tar -xvf fftw-3.3.8.tar.gz
cd fftw-3.3.8
./configure --enable-single --enable-shared --prefix=$ROOT
make -j
make install
cd .. 
rm -rf fftw-3.3.8 fftw-3.3.8.tar.gz
```
and 
```
git clone https://github.com/tuttieee/lear-gist-python
cd lear-gist-python
bash download-lear.sh
python setup.py build_ext -I $ROOT/include -L $ROOT/lib
python setup.py install
cd ..
```

## Data
### Images
I copied the some image files into the following. 
- `./data/att_obj_bbox_crop`: this is some crops from exp 12,27, and 91. Only included in the salk server. I guess not all crops are here, and only attended objects within the first three mins are included (but i don't remember clearly)?. The crop is based on bounding boxes provided by Andrei, so if you need the crop for other frames, you can make by yourself.
- `./data/sample_imgs`: I copied 50 sample images only for the purpose of this example. There's also the size information in `sample_imgs_size.csv` These are used in the example notebook later. 

The name (e.g., `1201-child_frame-1000_toy-12.jpg`) should be informative enough. Note that toy index start from 1 instead of 0. 

### Bounding Boxes
I already provide the cropped images but just for the record, I'm documenting the bounding box files from Andrei.
- exp12: `/data/aamatuni/code/postprocess_boxes/output/exp12/bbox_processed.json`
- exp15: `/data/aamatuni/code/postprocess_boxes/output/exp15/bbox_processed.json`
- exp27: `/data/aamatuni/code/postprocess_boxes/output/exp27bbox_processed.json`
- exp91: `/data/aamatuni/code/postprocess_boxes/output/exp91/bbox_processed.json`

You can also see `obj_frames` (e.g. `/data/aamatuni/code/postprocess_boxes/output/exp12/obj_frames`) directory along with each json and see some detected samples. 

Each detection is a dictionary with:
- `bbox` key corresponding to box coordinates (XYWH), 
- `category_id` corresponds to object class, 
- `fname` key corresponds to the name of the image frame that this detection refers to. These file names include metadata (e.g. `exp15_subj2018113023639_cam07_frame000000382.jpg`)

Please ask Andrei (aamatuni@indiana.edu) if you have more question on the object detection results as he did everything for this.

## Extract gist and make MDS plot. 
- see [`gist-mds-example.ipynb`](./ipython/gist-mds-example.ipynb). 
- If you don't know how to run the ipython notebook, you can use the python code [`gist-mds-example.py`](./code/gist-mds-example.py) with the following way.

```
cd code
/data/stsutsui/public/mds-vis/miniconda/bin/python gist-mds-example.py
```
This will save several pdf files in the `./results` directory.
