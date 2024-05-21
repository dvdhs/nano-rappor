# nano-rappor
Small, toy implementation of RAPPOR as final project for CMSC 25910.

## Project Layout
All relevent code is inside the `nanorappor` folder, which contains the
implementation code and the simulation code. The simulation code is
`example.py`, and the usage can be viewed with `python3 example.py --help`.
This is the code that generates the plots that are used in the paper, and each
subcommand will allow you to generate the plots for that trial. For example, to
run the trials varying the number of hash functions, and save the final plot in
the root directory, one can do
```
cd nanorappor
python3 example.py num-hashes-trial --output ../hashes.png
```

Note that the code is hardcoded to use 10 cores to multiprocess the trials,
since the code is pretty inefficient. 

`response.py` contains the class `RapporResponse`, which is intialized with
given RAPPOR parameters, and you can call `report_array` method to get the
final reporting array. It can also be memoized via pickling, which was the
goal. Because the file is not a module, you need to direct import or make
`__init__.py`. 

`decode.py` contains the code that decodes a list of `RapporResponse`'s for
simulation, function is called `decode_rappor_responses`. Read `example.py` to
see usages. 
