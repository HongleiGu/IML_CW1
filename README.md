# IML_CW1
Imperial Comp 60012 Intro to ML Coursework 1 (Decision Trees)

## How to run:

### jupyter notebook

since jupyter notebook is more visually appealing, I would reccomend using this

1. setup the link (optional)
in the terminal

```bash
ln -s /vol/lab/ml/intro2ml/bin/activate ~/intro2ml
```

2. go to the directory and activate the venv
in the terminal

```bash
cd <path-to-repo>
# if you have done step 1
source ~/intro2ml

# if you havent
source /vol/lab/ml/intro2ml/bin/activate
```

3. jupyter notebook
in the terminal

```bash
jupyter notebook
```

then use the browser and access localhost:8888/tree or wait for it to lead you automatically

double click on the ipynb file

4. run

select the cell shift+enter to run


### python file

the main.py is a exported python file of the ipynb with the same contents (markdown within may not be that readable, all the # % % cell notations are removed)

to run do:

1. setup the link (optional)
in the terminal

```bash
ln -s /vol/lab/ml/intro2ml/bin/activate ~/intro2ml
```

2. go to the directory and activate the venv
in the terminal

```bash
cd <path-to-repo>
# if you have done step 1
source ~/intro2ml

# if you havent
source /vol/lab/ml/intro2ml/bin/activate
```

3. run
```bash
python main.py
```