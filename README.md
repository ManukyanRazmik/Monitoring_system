# README #

```
docker build -t test .
docker run -d -it -p 8000:8000 --mount type=bind,source="$(pwd)"/target,target=/app
```

Change "$(pwd)"/target to the source that you want to mount.

Make sure you have `config` folder with `config.yaml` file.

Example of config.yaml
(the data should be in .rds format)
```
data_path: enroll_proj_data.rds
date_column: ENR_DATE
id_column: SUBJECT_ID
```

Make sure you have `data` folder with `.rds` data.