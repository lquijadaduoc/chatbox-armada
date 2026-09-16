import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="API Chatbot Convenios Armada Plus")

# Configuración de CORS para permitir peticiones del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. AUTENTICACIÓN Y SELECCIÓN DE MODELO (GROQ)
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def obtener_modelo_activo():
    """Busca dinámicamente si Qwen está disponible; de lo contrario usa Llama 3."""
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    modelo_fallback = "llama-3.1-8b-instant"
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            disponibles = [m["id"] for m in res.json().get("data", [])]
            qwen_models = [m for m in disponibles if "qwen" in m.lower()]
            return qwen_models[0] if qwen_models else modelo_fallback
    except Exception:
        pass
    return modelo_fallback

MODELO_USADO = obtener_modelo_activo()

# ==========================================
# 2. BASE DE CONOCIMIENTO OPTIMIZADA
# ==========================================
texto_completo = """
=== SECCIÓN 1: TÉRMINOS Y CONDICIONES GENERALES ===
El Servicio de Bienestar Social de la Armada, actuando como intermediario, suscribe convenios con distintas entidades comerciales a nivel nacional y regional, para favorecer la demanda de bienes y servicios del personal de la Armada y su familia.

Proceso de Adhesión (Suscripción):
- ¿Cómo adherirse o suscribirse?: El usuario debe completar el formulario de inscripción disponible en el portal web oficial www.sisbien.cl/conveniosarmadaplus. No requiere trámites presenciales, firmas físicas ni entrega de papeletas.
- La incorporación al Sistema de Convenios Armada Plus se hace efectiva dentro de los 7 días hábiles posteriores a su suscripción en línea.
- La suscripción o adhesión es por un tiempo indefinido, el que en ningún caso podrá ser inferior a un (1) año calendario contado desde la adhesión del usuario.

Condiciones de Pago, Valor de Cuotas y Mora:
- El VALOR, PRECIO o COSTO del aporte mensual depende de la situación del funcionario: para el personal activo corresponde a un porcentaje de su sueldo base; para el personal en retiro (pensión CAPREDENA) corresponde a un monto fijo; y para el personal de fondos propios u otras leyes corresponde a otro monto fijo establecido.
- El pago de esta suscripción se hace efectivo ÚNICAMENTE mediante descuento por mecanizado (descuento por papeleta para funcionarios activos y descuento en pensión CAPREDENA para funcionarios en retiro). No existen pagos por transferencia, efectivo ni tarjetas.
- El usuario en mora en 3 o más mensualidades consecutivas impagas será automáticamente dado de baja del Sistema, sin extinguir la obligación de pago.

=== SECCIÓN 2: PREGUNTAS FRECUENTES (FAQ) Y CASOS ESPECÍFICOS ===
P1: ¿Qué beneficios incluye el sistema Convenios Armada Plus y cuáles no?
R1: Estar suscrito permite acceso a todos los convenios comerciales, línea directa y promociones especiales. NO REQUIERE estar suscrito a Convenios Armada Plus para beneficios como: convenios habitacionales, tarjeta prepago fin de año y compras en unidades de negocios de Bienestar Social.

P2: ¿Quiénes se pueden adherir o suscribir al Sistema de Convenios Armada Plus?
R2: 
- Personal descrito en el art 2 del D.F.L. N° 1 (G) 1997: Oficiales, Gente de Mar, Tropa Profesional, Empleados Civiles, Personal a contrata y Reserva Llamado al Servicio Activo.
- Personal retirado de la Armada de Chile con pensión CAPREDENA: Oficiales, Gente de Mar y Empleados Civiles.
- Empleados públicos otras leyes, con contrato de trabajo vigente en Unidades dependientes de la Armada de Chile.

P3: ¿El sistema de Convenios Armada Plus tiene relación con las otras Fuerzas Armadas o de Orden?
R3: No. Es extensivo solo a convenios suscritos por la Armada de Chile y para sus funcionarios. No aplica para Ejército, FACH, Carabineros ni PDI.

P4: ¿A quién se le puede extender el beneficio de Convenios Armada Plus?
R4: 
- Extensión directa: Se extiende directamente al cónyuge. Asimismo, el personal en retiro (con pensión CAPREDENA) puede incorporar a su grupo familiar directo (cónyuge e hijos/as entre 15 y 24 años) acreditándolos con Libreta de Familia o certificado de matrimonio ante el Call Center. Los empleados públicos de otras leyes deben regularizarlo en su repartición.
- Terceros y otros familiares: Se pueden compartir los beneficios comerciales con otros familiares o terceros únicamente mientras el titular afiliado o su cónyuge se encuentren físicamente presentes al momento de hacer uso del convenio.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=90)
fragmentos = text_splitter.split_text(texto_completo)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
base_de_datos = FAISS.from_texts(fragmentos, embeddings)
retriever = base_de_datos.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(model=MODELO_USADO, temperature=0.0)

# ==========================================
# 3. PROMPT CON ASOCIACIÓN DE CONCEPTOS
# ==========================================
prompt = ChatPromptTemplate.from_template("""
Eres el asistente virtual oficial de soporte del Sistema de Convenios Armada Plus, dependiente del Servicio de Bienestar Social de la Armada de Chile.

Tu labor es orientar basándote EXCLUSIVAMENTE en el CONTEXTO OFICIAL provisto.

REGLAS DE ATENCIÓN:
1. SINÓNIMOS IMPORTANTES Y NORMALIZACIÓN DE VOCABULARIO:: 
   - "Adherirse", "suscribirse", "inscribirse" y "unirse" significan lo mismo.
   - "Valor", "precio", "costo" o "cuánto sale" se refieren a las Condiciones de Pago y Valor de Cuotas.
   - Las palabras "cap", "convenio armada" y "cuota comercial" se refieren todas al "Sistema de Convenio Armada Plus".
   - REGLA ESTRICTA DE REDACCIÓN: Sin importar qué término informal use el usuario en su pregunta, TÚ SIEMPRE debes usar el nombre oficial "Convenio Armada Plus" en tus respuestas. Nunca uses las abreviaturas del usuario.
   - "Adherirse", "suscribirse", "inscribirse" y "unirse" significan lo mismo.
   - "Valor", "precio", "costo" o "cuánto sale" se refieren a las Condiciones de Pago.


2. PREGUNTAS TRAMPA Y REGLAS DE NEGOCIO:
   - Valor/Precio: Si preguntan cuánto cuesta, explica que depende de la situación: es un porcentaje del sueldo base para activos, un monto fijo para retiro/CAPREDENA, y un monto fijo distinto para fondos propios.
   - Permanencia: Mínimo 1 año calendario desde la adhesión.
   - Otras ramas (Ejército, FACH, PDI): No aplica para ellos, es exclusivo de la Armada.
   - Extensión del beneficio: Se extiende de forma directa al cónyuge y al grupo familiar acreditado (hijos/as entre 15 y 24 años para personal retiro CAPREDENA). A terceros o demás familiares solo se puede extender si el titular o su cónyuge están físicamente presentes.
   - Formas de pago: Solo descuento por mecanizado.
   - Formas de suscribirse: Si preguntan por la via o por la forma de como me puedo adherir al convenio, solo se puede hacer por el formulario en linea en la pagina https://www.sisbien.cl/conveniosarmadaplus o indicar que se tiene que ingresar a la pagina del convenio armada plus y adherirse de manera online.
3. DESCONOCIMIENTO: Si una consulta no puede responderse con el contexto (por ejemplo, si te piden el monto exacto en pesos), responde EXACTAMENTE esto:
   "Esa información específica no se encuentra detallada en mis registros de Convenios Armada Plus. Te sugiero consultar a los canales de atención (32 2848535 – 32 2848502)."
4. TONO: Formal, institucional y directo.

CONTEXTO OFICIAL:
{context}

CONSULTA: {question}

RESPUESTA OFICIAL:
""")

def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

cadena_rag = (
    {"context": retriever | formatear_documentos, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4. ESQUEMAS Y ENDPOINTS REST
# ==========================================
class ChatRequest(BaseModel):
    pregunta: str

class ChatResponse(BaseModel):
    respuesta: str
    modelo: str

@app.get("/")
def health_check():
    return {"status": "online", "modelo_activo": MODELO_USADO}

@app.post("/api/chat", response_model=ChatResponse)
def endpoint_chat(req: ChatRequest):
    if not req.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    try:
        resultado = cadena_rag.invoke(req.pregunta)
        return ChatResponse(respuesta=resultado, modelo=MODELO_USADO)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))