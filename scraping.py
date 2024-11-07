from bs4 import BeautifulSoup
import requests
import pandas as pd

def obtener_datos():
    website = "https://listado.mercadolibre.com.co/inmuebles/casas/venta/antioquia/medellin/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    }
    resultado = requests.get(website, headers=headers)
    content = resultado.text
    soup = BeautifulSoup(content, "html.parser")
    anuncios = soup.find_all("div", class_="poly-card__content")

    # Crear listas para almacenar la información
    titulos, precios, nro_habitaciones, nro_banos, metros_cuadrados, barrios, ciudades, departamentos = [], [], [], [], [], [], [], []

    for anuncio in anuncios:
        # Extraer datos del anuncio
        titulo_elem = anuncio.find("span", class_="poly-component__headline")
        titulo = titulo_elem.get_text() if titulo_elem else "No disponible"
        precio_elem = anuncio.find("span", class_="andes-money-amount andes-money-amount--cents-superscript")
        precio = precio_elem["aria-label"] if precio_elem else "No disponible"
        
        # Extraer atributos
        atributos = anuncio.find("div", class_="poly-component__attributes-list")
        num_habitaciones = "No disponible"
        num_banos = "No disponible"
        metros_cuadrados_val = "No disponible"
        
        if atributos:
            detalles = [item.get_text() for item in atributos.find_all("li")]
            for detalle in detalles:
                if "habitaciones" in detalle:
                    num_habitaciones = detalle.split()[0]
                elif "baños" in detalle:
                    num_banos = detalle.split()[0]
                elif "m²" in detalle:
                    metros_cuadrados_val = detalle.split()[0]
        
        # Extraer ubicación
        ubicacion_elem = anuncio.find("span", class_="poly-component__location")
        if ubicacion_elem:
            ubicacion = ubicacion_elem.get_text()
            ubicacion_parts = ubicacion.split(", ")
            barrio = ubicacion_parts[0] if len(ubicacion_parts) > 0 else "No disponible"
            ciudad = ubicacion_parts[1] if len(ubicacion_parts) > 1 else "No disponible"
            departamento = ubicacion_parts[2] if len(ubicacion_parts) > 2 else "No disponible"
        else:
            barrio, ciudad, departamento = "No disponible", "No disponible", "No disponible"

        # Agregar datos a las listas
        titulos.append(titulo)
        precios.append(precio)
        nro_habitaciones.append(num_habitaciones)
        nro_banos.append(num_banos)
        metros_cuadrados.append(metros_cuadrados_val)
        barrios.append(barrio)
        ciudades.append(ciudad)
        departamentos.append(departamento)

    # Crear DataFrame
    data = {
        "Título": titulos,
        "Precio": precios,
        "NRO_HABITACIONES": nro_habitaciones,
        "NRO_BAÑOS": nro_banos,
        "METROS_CUADRADOS": metros_cuadrados,
        "BARRIO": barrios,
        "CIUDAD": ciudades,
        "DEPARTAMENTO": departamentos
    }
    df = pd.DataFrame(data)
    df.to_csv("data/casas_medellin.csv", index=False, encoding="utf-8-sig")
    
    return df
