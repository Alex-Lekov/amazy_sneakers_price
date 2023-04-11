# AI Estimating the price of amazy sneakers

<p align="center">
  <a href="" rel="noopener">
 <img width=400px height=200px src="https://pbs.twimg.com/media/FTjCaINXoAUQMUx.jpg" alt="Bot logo"></a>
</p>

<h3 align="center">Based on ML with FastAPI Serving </h3>

<div align="center">

  [![Python](https://img.shields.io/badge/python-v3.10-blue.svg)]()

</div>

## Status: Archive

Because of the decline in activity on the market Amazy project had to Close. The project is considered a success because we were able to make a profit from it.


---

## General info

> With the AI we predict the price of a sneaker on the market, find and redeem sneakers that are lower than the market. We make money on their resale.

marketplace: [https://go.amazy.io/marketplace](https://go.amazy.io/marketplace)

- To train the model we use the sales history collected from the blockchain (you can see the latest version of the model and its metrics in the [notebooks](./notebooks/) folder)
- FastAPI is used to get predicates online [http://0.0.0.0:8003/docs#/](http://0.0.0.0:8003/docs#/)

---
## Technologies
Project is created with:
* python: 3.10
* catboost: 1.0.6
* fastapi: 0.85.0
* docker: 20.10.16
* docker-compose: 1.29.2


## Installation 
with Docker Compose :whale:

1. setup [config_prod.yml](./app/config_prod.yml)
2. run
```bash
> docker-compose build
> docker-compose up
```
3. Open [http://0.0.0.0:8003](http://0.0.0.0:8003) FastAPI Swagger UI

## CHANGELOG
[CHANGELOG](./CHANGELOG.md)

