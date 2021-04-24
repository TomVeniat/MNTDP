# MNTDP
/!\ Documentation in progress, do not hesitate to open an issue for questions.

### Installation:

#### Dependencies
```bash
conda create -n MNTDP python=3.8
conda activate MNTDP
pip install -r requirements.txt
```

Now let's get Mongo and start a server:

```bash
mkdir -p /checkpoint/${USER}/mongo/{db,logs}
cd /checkpoint/${USER}/mongo 
wget -O - https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1804-4.2.0.tgz | tar -xzvf -  
./mongodb-linux-x86_64-ubuntu1804-4.2.0/bin/mongod --dbpath /checkpoint/${USER}/mongo/db --logpath /checkpoint/${USER}/mongo/logs/mongodb.log --fork
```

Which should give the following output: `child process started successfully, parent exiting`
Mongo is used with [Sacred](https://github.com/IDSIA/sacred) to keep track of the results.

The next step is to start Ray's head node:
```bash
ray start --head --redis-port 6381
```
by default, ray will use all gpus available.

and then the experiment:
```bash
python run.py with configs/streams/s_plus.yaml
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

### Additional configurations
Specific configurations for Mongo and Visdom can be provided by editing the corresponding file in the resources folder.
