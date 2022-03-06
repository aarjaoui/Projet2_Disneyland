from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

tags_metadata = [
    {
        "name": "status",
        "description": "Retourner status de service.",
    },
    {
        "name": "auth",
        "description": "Vérifier l'authentification de l'utilisateur.",
    },
]


app = FastAPI(openapi_tags=tags_metadata)

security = HTTPBasic()


@app.get("/status", tags=["status"])
def get_status():
    return {"Service is OK": 1}

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username == "alice" and credentials.password == "wonderland":
        return {"Hello ": credentials.username}
    elif credentials.username == "bob" and credentials.password == "builder":
        return {"Hello ": credentials.username}
    elif credentials.username == "clementine" and credentials.password == "mandarine":
        return {"Hello ": credentials.username}
    else:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect user or password",
            headers={"WWW-Authenticate": "Basic"},
        )

@app.get("/auth", tags=["auth"])
def read_current_user(username: str = Depends(get_current_username)):
    return {"username": username}

@app.get("/score/modele1")
def ger_score_modele1(username: str = Depends(get_current_username)):
    return {"score": 1}

@app.get("/score/modele2")
def ger_score_modele1(username: str = Depends(get_current_username)):
    return {"score": 0}
