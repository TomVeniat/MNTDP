# Lileb Training

This project contains different

### Installation:

#### Dependencies
```bash
conda create -n MNTDP python=3.8
conda activate MNTDP
pip install -r requirements.txt
#conda install pygraphviz
```

Now let's get Mongo and start a server:

```bash
mkdir -p /checkpoint/${USER}/mongo/{db,logs}
cd /checkpoint/${USER}/mongo 
wget -O - https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1804-4.2.0.tgz | tar -xzvf -  
./mongodb-linux-x86_64-ubuntu1804-4.2.0/bin/mongod --dbpath /checkpoint/${USER}/mongo/db --logpath /checkpoint/${USER}/mongo/logs/mongodb.log --fork
```

Which should give us the following output: `child process started successfully, parent exiting`

We can now run a test experiment:

```bash
cd -
python run.py with configs/test_debug.yaml
```

If everything runs smoothly, we can now launch ray:
First, we need to launch Ray's head node:
```bash
ray start --head --redis-port 6382
```
and then the experiment:
```bash
python run.py with configs/streams/transfer.yaml
```

the different files in the `config/streams/` directory corresponds to the streams of the CTrL benchmark.


To stop the mongo server:

```bash
/checkpoint/${USER}/mongo/mongodb-linux-x86_64-ubuntu1804-4.2.0/bin/mongod --dbpath /checkpoint/${USER}/mongo/db --shutdown
```

To stop Ray head node:

```bash
ray stop
```