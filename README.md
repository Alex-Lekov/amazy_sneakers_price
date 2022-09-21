# AI Estimating the price of amazy sneakers

<p align="center">
  <a href="" rel="noopener">
 <img width=400px height=200px src="https://pbs.twimg.com/media/FTjCaINXoAUQMUx.jpg" alt="Bot logo"></a>
</p>

<h3 align="center">Based on ML with FastAPI Serving </h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![Python](https://img.shields.io/badge/python-v3.10-blue.svg)]()

</div>

---

## General info
Предсказываем стоисть кросовка на маркете [https://go.amazy.io/marketplace](https://go.amazy.io/marketplace)
- Для обучения модели используем историю продаж собранную с блокчейн (посмотреть последнию версию модели и ее метрики можно в папке [notebooks](./notebooks/))
- Для получение предиктов в онлайне используеться FastAPI [http://0.0.0.0:8003/docs#/](http://0.0.0.0:8003/docs#/)

## Technologies
Project is created with:
* python: 3.10
* catboost: 1.0.6
* fastapi: 0.85.0
* docker: 20.10.16
* docker-compose: 1.29.2

## Installation with Docker Compose :whale:
1. setup [config_prod.yml](./app/config_prod.yml)
2. run
```bash
> docker-compose build
> docker-compose up
```
3. Open [http://0.0.0.0:8003](http://0.0.0.0:8003) FastAPI Swagger UI

# CHANGELOG
[CHANGELOG](./CHANGELOG.md)

# TODO

-   [ ] Добавить признак минимальной цены на текущем рынке

-   [ ] Дообучить модель оценки профитности сделки до приемлимого уровня ошибки


