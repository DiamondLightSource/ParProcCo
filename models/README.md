Get OpenAPI schema from endpoint:
```
$ curl -H X-SLURM-USER-NAME:${USER} -H X-SLURM-USER-TOKEN:${SLURM_TOKEN) "${SLURM_REST_URL}/openapi" -o slurm-rest.out
```
Filter and rename schema and refs to remove version
```
$ python generate_models.py slurm-rest.out
```
Run datamodel generating script
```
$ datamodel-codegen --input slurm-api.yaml --target-python-version 3.11 --use-schema-description --use-field-description --output-model-type pydantic_v2.BaseModel --use-union-operator --use-standard-collections --field-constraints --output slurm_rest.py
```
