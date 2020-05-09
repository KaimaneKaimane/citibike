clean-pycache:
	find . -type d -name __pycache__ -exec rm -r {} \+

notebook-build: clean-pycache
	docker build . -f ./docker/notebook/Dockerfile -t citybike/notebook:latest --cache-from citybike/notebook:latest

notebook-run:
	docker run -it -p 8888:8888 citybike/notebook:latest

notebook-lock:
	cd docker/notebook/; pipenv lock

base: clean-pycache
	docker build . -f ./docker/project/Dockerfile -t citybike/base:latest --cache-from citybike/base:latest --build-arg PROJECT=citibike

base-lock:
	cd docker/project/; pipenv lock

api:
	docker-compose -f docker-compose.yml up