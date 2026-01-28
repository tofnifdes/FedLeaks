

## Prerequisites
The code was tested using Python (3.8)

**Packages**: Install using `pip install -r requirements.txt`

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Project Setup

Start by cd into your desired AdNN and Dataset Config Folder.

## For Benign training run

```shell
./benigntraining.sh
```

## For Malicious training run

You would need to specify your target attribute in the fedleaks.sh file, it defaults to classLabel
```shell
./fedleaks.sh
```
