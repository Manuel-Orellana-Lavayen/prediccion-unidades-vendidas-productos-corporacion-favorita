# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Función Estilos Gŕaficos plotly

    Este módulo tiene una función que aplica un estilo a los gráficos creados con Plotly
"""

def estilos_plotly(

    figura,
    titulo: str,
    color_titulo: str,
    titulo_leyenda: str,
    color_texto_leyenda: str,
    fondo_transparente: bool,
    leyenda: bool,
    json: bool = False,
):
    """Aplica un estilo visual estandarizado a una figura de Plotly.

        Configura el título, fondo, leyenda

        Opcionalmente, convierte la figura resultante a formato JSON.

        Args:
            figura: Figura de Plotly a la que se aplicarán los estilos.
            titulo (str): Texto que se mostrará como título de la figura.
            color_titulo (str): Color del título.
            titulo_leyenda (str): Título mostrado en la leyenda.
            color_texto_leyenda (str): Color del texto de la leyenda.
            fondo_transparente (bool): Indica si el fondo de la figura debe ser transparente.
            leyenda (bool): Indica si se debe mostrar la leyenda.
            json (bool): Indica si la figura debe convertirse a formato JSON.

        Returns:
            La figura de Plotly con los estilos aplicados. Si `json` es True, devuelve la representación JSON de la figura.
        """
    # Agrupamos todas las propiedades del título dentro de un único diccionario raíz
    figura.update_layout(
        title=dict(
            text=titulo,
            x=0.5,
            xanchor="center",
            font=dict(color=color_titulo, size=20),
        )
    )

    if fondo_transparente:
        figura.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )

    if leyenda:
        figura.update_layout(
            showlegend=True,
            legend=dict(
                title=dict(text=titulo_leyenda),
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=12, color=color_texto_leyenda),
            ),margin=dict(r=180)
        )

    if json:
        figura = figura.to_json()

    return figura