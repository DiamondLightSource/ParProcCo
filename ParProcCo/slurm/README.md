Get OpenAPI schema from endpoint:
```
$ curl -H X-SLURM-USER-NAME:${USER} -H X-SLURM-USER-TOKEN:${SLURM_TOKEN) "${SLURM_REST_URL}/openapi" -o slurm-rest.json
```
Filter and rename schema and refs to remove version
```
$ python generate_models.py slurm-rest.json
$ python generate_models.py -d slurm-rest.json slurmdb-rest.yaml
```
Run datamodel generating script
```
$ datamodel-codegen --input slurm-rest.yaml --target-python-version 3.11 --use-schema-description --use-field-description --output-model-type pydantic_v2.BaseModel --use-union-operator --use-standard-collections --field-constraints --use-double-quotes --output slurm_rest.py
$ datamodel-codegen --input slurmdb-rest.yaml --target-python-version 3.11 --use-schema-description --use-field-description --output-model-type pydantic_v2.BaseModel --use-union-operator --use-standard-collections --field-constraints --use-double-quotes --output slurmdb_rest.py
```

