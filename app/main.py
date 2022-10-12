####################################### IMPORT #################################
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.responses import FileResponse
#from fastapi.middleware.cors import CORSMiddleware
import uuid
import pandas as pd
import numpy as np
from typing import Union
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger
import sys
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import base64
import yaml
import io 


####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add(
    "log.log", rotation="1 MB", level="DEBUG", compression="zip"
)

####################################### SETUP #################################

####### LOAD CONFIG ##################################
with open("config_prod.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

MODEL_DIR = config['MODEL_DIR']
VERSION = config['VERSION']

####################################### data_quality setup #################################

data_quality_expected_range_dict = {
    'sneaker':
        {
        #'is_type': {'genesis': bool},
        'is_in':
            {
            'rarity': ['common', 'uncommon', 'rare'],
            'sneaker_type': ['ranger', 'sprinter', 'hiker', 'coacher'],
            },
        'min_max':
            {
            'level': [0, 30],
            'base_performance': [0, 40],
            'base_fortune': [0, 40],
            'base_joy': [0, 40],
            'base_durability': [0, 40],
            'performance': [0, 500],
            'fortune': [0, 300],
            'joy': [0, 300],
            'durability': [0, 300],
            'mint': [0, 7],
            #'buy_count_12H': [0, 1000],
            'buy_count_24H': [0, 2000],
            'sell_count_24H': [0, 4000],
            'cancel_count_24H': [0, 3000],
            #'token_all_activity_3H': [0, 30],
            #'token_sell_activity_6H': [0, 30],
            },
        }
}


###############################################################################


app = FastAPI(
    title="Estimating the price of amazy sneakers",
    version=VERSION,
    description="Based on ML with FastAPI Serving ⚡",
)


class RequestModel(BaseModel):
    token_id: int = Field(
        example=22259,
        description="token_id: int",
    )
    item_type: str = Field(
        example='sneakers', description="item_type: sneakers, box"
    )
    rarity: str = Field(
        example='common', description="rarity: common, uncommon, rare"
    )
    sneaker_type: Union[str, None] = Field(
        example='hiker',
        description="sneaker_type: ranger, sprinter, hiker, coacher",
    )
    genesis: bool = Field(
        example=False,
        description="genesis: bool",
    )
    level: Union[int, None] = Field(
        example=5,
        description="level: int",
    )
    base_performance: Union[float, None] = Field(
        example=3.0,
        description="base_performance: float",
    )
    base_fortune: Union[float, None] = Field(
        example=2.7,
        description="base_fortune: float",
    )
    base_joy: Union[float, None] = Field(
        example=7.3,
        description="base_joy: float",
    )
    base_durability: Union[float, None] = Field(
        example=3.5,
        description="base_durability: float",
    )
    performance: Union[float, None] = Field(
        example=16.5,
        description="performance: float",
    )
    fortune: Union[float, None] = Field(
        example=4.2,
        description="fortune: float",
    )
    joy: Union[float, None] = Field(
        example=10.8,
        description="joy: float",
    )
    durability: Union[float, None] = Field(
        example=8.5,
        description="durability: float",
    )
    mint: Union[int, None] = Field(
        example=2,
        description="mint: int",
    )
    parent1_sneaker_type: Union[str, None] = Field(
        example='hiker',
        description="parent1_sneaker_type: ranger, sprinter, hiker, coacher. Or null if genesis or item_type sneakers",
    )
    parent2_sneaker_type: Union[str, None] = Field(
        example='hiker',
        description="parent2_sneaker_type: ranger, sprinter, hiker, coacher. Or null if genesis or item_type sneakers",
    )
    wallet_from_buy_count: Union[int, None] = Field(
        example=0,
        description="wallet_from_buy_count: int",
    )
    wallet_from_all_count: Union[int, None] = Field(
        example=1,
        description="wallet_from_all_count: int",
    )
    buy_count_12H: Union[int, None] = Field(
        example=97,
        description="buy_count_12H: int",
    )
    buy_count_24H: Union[int, None] = Field(
        example=191,
        description="buy_count_24H: int",
    )
    buy_count_48H: Union[int, None] = Field(
        example=191,
        description="buy_count_48H: int",
    )
    sell_count_12H: Union[int, None] = Field(
        example=430,
        description="sell_count_12H: int",
    )
    sell_count_24H: Union[int, None] = Field(
        example=430,
        description="sell_count_24H: int",
    )
    sell_count_48H: Union[int, None] = Field(
        example=430,
        description="sell_count_48H: int",
    )
    cancel_count_12H: Union[int, None] = Field(
        example=238,
        description="cancel_count_12H: int",
    )
    cancel_count_24H: Union[int, None] = Field(
        example=238,
        description="cancel_count_24H: int",
    )
    cancel_count_48H: Union[int, None] = Field(
        example=238,
        description="cancel_count_48H: int",
    )
    min_price_all_24H: Union[float, None] = Field(
        example=0.6,
        description="min_price_all_24H: float",
    )
    min_price_by_rarity_genesis_type_level_mint_24H: Union[float, None] = Field(
        example=0.7,
        description="min_price_by_rarity_genesis_type_level_mint_24H: float or null",
    )
    min_price_by_rarity_genesis_type_level_mint_48H: Union[float, None] = Field(
        example=0.7,
        description="min_price_by_rarity_genesis_type_level_mint_48H: float or null",
    )
    min_price_by_rarity_genesis_type_level_mint_72H: Union[float, None] = Field(
        example=0.7,
        description="min_price_by_rarity_genesis_type_level_mint_72H: float or null",
    )
    min_price_by_rarity_genesis_type_24H: Union[float, None] = Field(
        example=0.7,
        description="min_price_by_rarity_genesis_type_24H: float or null",
    )
    min_price_by_rarity_genesis_type_48H: Union[float, None] = Field(
        example=0.7,
        description="min_price_by_rarity_genesis_type_48H: float or null",
    )
    min_price_by_rarity_genesis_type_72H: Union[float, None] = Field(
        example=0.7,
        description="min_price_by_rarity_genesis_type_72H: float or null",
    )
    token_all_activity_3H: Union[int, None] = Field(
        example=1,
        description="token_all_activity_3H: int",
    )
    token_sell_activity_6H: Union[int, None] = Field(
        example=1,
        description="token_sell_activity_6H: int",
    )
    price_bnb: float = Field(
        example=277.87,
        description="price_bnb: float",
    )
    price_azy: float = Field(
        example=0.087,
        description="price_azy: float",
    )
    price_amt: float = Field(
        example=0.55,
        description="price_amt: float",
    )
    wallet_first_sneaker_time: Union[int, None] = Field(
        example=2221030,
        description="wallet_first_sneaker_time: int Время когда появился первый кроссовок на кошельке",
    )
    time_ownership: Union[int, None] = Field(
        example=222103,
        description="time_ownership: int Сколько прошло времени с момента когда появился данный кроссовок на кошельке (если была отмена продажи считается с этого времени)",
    )
    wallet_box_mint: Union[int, None] = Field(
        example=4,
        description="wallet_box_mint: int Сколько боксов сминчено на кошельке",
    )
    wallet_sneaker_mint: Union[int, None] = Field(
        example=4,
        description="wallet_sneaker_mint: int Сколько боксов открыто на кошельке",
    )
    time_level_up: Union[int, None] = Field(
        example=900,
        description="time_level_up: int Сколько времени потрачено, что бы прокачать до текущего лвл",
    )
    time_level_up_for_mint: Union[int, None] = Field(
        example=1800,
        description="time_level_up_for_mint: int Сколько времени нужно, что бы прокачать 2 кроссовка до 5 лвл, что бы сделать минт",
    )
    base_mint_price_amt: Union[float, None] = Field(
        example=300.0,
        description="base_mint_price_amt: float Сколько нужно AMT для минта, при скрещивании двух пар такой же редкости",
    )
    base_mint_price_azy: Union[float, None] = Field(
        example=200.0,
        description="base_mint_price_azy: float Сколько нужно AZY для минта, при скрещивании двух пар такой же редкости",
    )
    base_mint_price_bnb: Union[float, None] = Field(
        example=0.67,
        description="base_mint_price_bnb: float Перевод BNB для минта, при скрещивании двух пар такой же редкости",
    )
    price: Union[float, None] = Field(
        example=0.89,
        description="price: float bnb",
    )
    explain_models: bool = Field(
        example=False,
        description="explain_models: bool",
    )


class ResponseModel(BaseModel):
    prediction_Id: str
    item_type: str
    token_id: int
    model_predict: Union[float, None]
    base_model_predict: float
    profit_model_predict: Union[float, None]
    model_predict_explain_img: Union[str, None]
    base_model_predict_explain_img: Union[str, None]
    profit_model_predict_explain_img: Union[str, None]



def flag_none_in_market_data(data_dict: dict):
    flag = True

    for key in [
        #'wallet_from_buy_count', 
        #'wallet_from_all_count',
        #'buy_count_12H',
        'buy_count_24H',
        'sell_count_24H',
        'cancel_count_24H',
        'min_price_all_24H',
        #'token_all_activity_3H',
        #'token_sell_activity_6H',
        'price_bnb',
        #'price_azy',
        #'price_amt',
        #'price',
        ]:
        if data_dict[key] is None:
            flag=False
    #print(flag)
    return flag


def get_explain_model_img_base64(model, data: pd.Series) -> base64:
    """Get explain model img in b64encode

    Args:
        model (_type_): Catboost model
        data (pd.Series): data

    Returns:
        base64: image b64 encodet
    """
    pic_IObytes = io.BytesIO()

    sample = pd.DataFrame(data[model.feature_names_]).T
    sample = sample.fillna(np.nan)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.force_plot(explainer.expected_value, shap_values[0,:], sample, show=False, matplotlib=True).savefig(pic_IObytes, format="png", dpi=150, bbox_inches='tight')

    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return pic_hash


def sneaker_test_data_quality(sneaker_dict: dict):
    """Checking data for compliance

    Args:
        sneaker_dict (_type_): data dict

    Raises:
        HTTPException
    """
    # is_in
    dq = data_quality_expected_range_dict['sneaker']['is_in']
    for feature in dq.keys():
        if sneaker_dict[feature] not in dq[feature]:
            raise HTTPException(status_code=422, detail=f'{feature} not in data_quality_expected_range_dict. given {sneaker_dict[feature]} expected: {dq[feature]}')

    # min_max
    dq = data_quality_expected_range_dict['sneaker']['min_max']
    for feature in dq.keys():
        if sneaker_dict[feature] < dq[feature][0]:
            raise HTTPException(status_code=422, detail=f'{feature} less then expected. get: {sneaker_dict[feature]} | expected: > {dq[feature][0]}')

        if sneaker_dict[feature] > dq[feature][1]:
            raise HTTPException(status_code=422, detail=f'{feature} more then expected. get: {sneaker_dict[feature]} | expected: < {dq[feature][1]}')


def box_test_data_quality(box_dict: dict):
    """Checking data for compliance

    Args:
        box_dict (_type_): box data dict

    Raises:
        HTTPException: Value Error: genesis sneaker no parents!
        HTTPException: Value Error: parent1_sneaker_type type
        HTTPException: Value Error: parent2_sneaker_type type
    """
    sneaker_types = ['ranger', 'sprinter', 'hiker', 'coacher']

    if box_dict['genesis']:
        if box_dict['parent1_sneaker_type'] in sneaker_types or box_dict['parent2_sneaker_type'] in sneaker_types:
            raise HTTPException(status_code=422, detail="Value Error: genesis sneaker no parents! pass 'null' in parent if box genesis")
    else:
        if box_dict['parent1_sneaker_type'] not in sneaker_types:
            raise HTTPException(status_code=422, detail=f"Value Error: parent1_sneaker_type type, given: {box_dict['parent1_sneaker_type']} expected: ranger, sprinter, hiker, coacher")
        elif box_dict['parent2_sneaker_type'] not in sneaker_types:
            raise HTTPException(status_code=422, detail=f"Value Error: parent2_sneaker_type type, given: {box_dict['parent2_sneaker_type']} expected: ranger, sprinter, hiker, coacher")


def sneaker_price_prediction(data_dict: dict):
    """main sneaker price predictions

    Args:
        data_dict: input data

    Returns:
        dict: predict & data
    """
    explain_model_img = None
    explain_base_model_img = None
    explain_profit_model_img = None
    model_predict_price = None
    base_model_predict_price = None
    profit_model_predict_price = None

    # Base Model
    base_model_predict_price = sneaker_base_model.predict(pd.Series(data_dict)[sneaker_base_model.feature_names_])
    base_model_predict_price = np.round(base_model_predict_price, 3)

    # Model V2
    if flag_none_in_market_data(data_dict):
        sneaker_test_data_quality(data_dict)
        data_dict['sum_activity_24H'] = data_dict['buy_count_24H'] + data_dict['sell_count_24H'] + data_dict['cancel_count_24H']
        data_dict['sells_activity_24H'] = data_dict['sell_count_24H'] / data_dict['buy_count_24H']

        sample = pd.Series(data_dict).fillna(np.nan)
        model_predict_price = sneaker_model.predict(sample[sneaker_model.feature_names_])
        model_predict_price = np.round(model_predict_price, 3)

        # profit_model
        if data_dict['price'] is not None:
            sample['predict'] = model_predict_price
            sample['predict_base'] = base_model_predict_price
            sample['profit_by_predict_model'] = np.round((sample['predict'] - ((sample['predict']/100) * 5)) - sample['price'], 3)
            sample['profit_by_predict_base_model'] = np.round((sample['predict_base'] - ((sample['predict_base']/100) * 5)) - sample['price'], 3)

            profit_model_predict_price = sneaker_profit_model.predict(sample[sneaker_profit_model.feature_names_])
            profit_model_predict_price = np.round(profit_model_predict_price, 3)

    # Report
    result = {
        "prediction_Id": str(uuid.uuid1()), 
        "item_type": data_dict["item_type"],
        "token_id": data_dict["token_id"],
        "model_predict": model_predict_price, 
        "base_model_predict": base_model_predict_price,
        "profit_model_predict": profit_model_predict_price,
        }
    logger.info(result)

    # Explain image
    if data_dict['explain_models']:
        explain_base_model_img = get_explain_model_img_base64(sneaker_base_model, sample)
        if flag_none_in_market_data(data_dict):
            explain_model_img = get_explain_model_img_base64(sneaker_model, sample)
            if data_dict['price'] is not None:
                explain_profit_model_img = get_explain_model_img_base64(sneaker_profit_model, sample)

        logger.info(f'Make new explainer_img')

        result['model_predict_explain_img'] = explain_model_img
        result['base_model_predict_explain_img'] = explain_base_model_img
        result['profit_model_predict_explain_img'] = explain_profit_model_img

    return result


def box_price_prediction(box_dict: dict):
    """main box price predictions

    Args:
        box_dict: input data

    Returns:
        dict: predict & data
    """
    explain_model_img = None
    explain_base_model_img = None
    model_predict_price = None
    base_model_predict_price = None

    box_test_data_quality(box_dict)

    data_tmp = pd.Series(index = box_model.feature_names_, dtype='int64')
    data_tmp['rarity'] = box_dict['rarity']
    data_tmp['genesis'] = box_dict['genesis']

    if flag_none_in_market_data(box_dict):
        for feature in box_dict.keys():
            data_tmp[feature] = box_dict[feature]

        data_tmp['sum_activity_24H'] = box_dict['buy_count_24H'] + box_dict['sell_count_24H'] + box_dict['cancel_count_24H']
        data_tmp['sells_activity_24H'] = box_dict['sell_count_24H'] / box_dict['buy_count_24H']

    if not box_dict['genesis']:
        feature_name = f"{box_dict['parent1_sneaker_type']}_{box_dict['parent2_sneaker_type']}"
        if feature_name not in box_model.feature_names_:
            feature_name = f"{box_dict['parent2_sneaker_type']}_{box_dict['parent1_sneaker_type']}"
        data_tmp[feature_name] = 1
    #print(data)
    if flag_none_in_market_data(box_dict):
        model_predict_price = box_model.predict(data_tmp[box_model.feature_names_])
        model_predict_price = np.round(model_predict_price, 3)

    base_model_predict_price = box_base_model.predict(data_tmp[box_base_model.feature_names_])
    base_model_predict_price = np.round(base_model_predict_price, 3)

    result = {
        "prediction_Id": str(uuid.uuid1()),
        "item_type": box_dict["item_type"],
        "token_id": box_dict["token_id"],
        "model_predict": model_predict_price, 
        "base_model_predict": base_model_predict_price,
        }
    logger.info(result)

    # Explain image
    if box_dict['explain_models']:
        if flag_none_in_market_data(box_dict):
            explain_model_img = get_explain_model_img_base64(box_model, data_tmp)
        explain_base_model_img = get_explain_model_img_base64(box_base_model, data_tmp)
        logger.info('Make new Box explainer_img')

        result['model_predict_explain_img'] = explain_model_img
        result['base_model_predict_explain_img'] = explain_base_model_img

    return result

############################# Requests ##########################################################

@app.post("/predict_price", response_model=ResponseModel)
async def price_prediction(body: RequestModel):
    """main price predictions ResponseModel

    Args:
        body (RequestModel): input data

    Returns:
        dict: predict & data
    """
    data_dict = body.dict()
    logger.info(data_dict)

    if data_dict['item_type'] == 'sneakers':
        result = sneaker_price_prediction(data_dict)
    elif data_dict['item_type'] == 'box':
        result = box_price_prediction(data_dict)
    else:
        raise HTTPException(status_code=422, detail=f"Value Error item_type type, given: {data_dict['item_type']} expected: sneakers, box")

    return result


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


########################## MAIN ###########################################################
###########################################################################################

if __name__ == "__main__":
    ####################### Models ###########################################
    # Sneaker MODELS
    sneaker_model = CatBoostRegressor()      # parameters not required.
    sneaker_model.load_model(f'{MODEL_DIR}sneaker_model_{VERSION}.model')

    sneaker_base_model = CatBoostRegressor()      # parameters not required.
    sneaker_base_model.load_model(f'{MODEL_DIR}sneaker_base_model_{VERSION}.model')

    # BOX MODELS
    box_model = CatBoostRegressor()      # parameters not required.
    box_model.load_model(f'{MODEL_DIR}box_model_{VERSION}.model')

    box_base_model = CatBoostRegressor()      # parameters not required.
    box_base_model.load_model(f'{MODEL_DIR}box_base_model_{VERSION}.model')

    # Profit Model
    sneaker_profit_model = CatBoostRegressor()      # parameters not required.
    sneaker_profit_model.load_model(f'{MODEL_DIR}sneaker_profit_model_{VERSION}.model')

    ######################## START ###########################################
    uvicorn.run(app, host=config['HOST'], port=config['PORT'])