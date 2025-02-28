import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import json
import plotly.figure_factory as ff

# Charger les données (à remplacer par votre méthode de chargement)
df = pd.read_csv('dataset_imputed.csv')

# Préparer les données
disorders = [
    'Schizophrenia (%)', 
    'Bipolar disorder (%)', 
    'Eating disorders (%)',
    'Anxiety disorders (%)', 
    'Drug use disorders (%)', 
    'Depression (%)',
    'Alcohol use disorders (%)'
]

disorder_names = {
    'Schizophrenia (%)': 'Schizophrénie', 
    'Bipolar disorder (%)': 'Trouble bipolaire', 
    'Eating disorders (%)': 'Troubles alimentaires',
    'Anxiety disorders (%)': 'Troubles anxieux', 
    'Drug use disorders (%)': 'Troubles liés aux drogues', 
    'Depression (%)': 'Dépression',
    'Alcohol use disorders (%)': 'Troubles liés à l\'alcool'
}

colors = {
    'Schizophrenia (%)': '#1f77b4',
    'Bipolar disorder (%)': '#ff7f0e',
    'Eating disorders (%)': '#2ca02c',
    'Anxiety disorders (%)': '#d62728',
    'Drug use disorders (%)': '#9467bd',
    'Depression (%)': '#8c564b',
    'Alcohol use disorders (%)': '#e377c2'
}

# Prétraitement des données
df['Year'] = df['Year'].astype(int)

# Filtrer les années de 1970 à 2019
df = df[(df['Year'] >= 1970) & (df['Year'] <= 2019)]

# Agréger les données par an pour la vue mondiale
world_yearly_avg = df.groupby('Year')[disorders].mean().reset_index()

# Obtenir la liste des pays et des années uniques
countries = sorted(df['Entity'].unique())
years = sorted(df['Year'].unique())

# Initier l'application Dash avec un thème Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Définition de l'interface utilisateur du tableau de bord
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Tableau de Bord des Tendances Mondiales en Santé Mentale", 
                         className="text-center my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filtres", className="card-header"),
                dbc.CardBody([
                    html.H6("Sélectionnez la plage d'années:"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=1970,
                        max=2019,
                        step=1,
                        marks={i: str(i) for i in range(1970, 2020, 10)},
                        value=[1990, 2019]
                    ),
                    html.Div(id='year-range-display', className="mt-2"),
                    
                    html.H6("Sélectionnez les pays:", className="mt-3"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in countries],
                        value=['Benin', 'France', 'United States', 'China', 'India','Nigeria','Niger',],
                        multi=True,
                        placeholder="Sélectionnez des pays"
                    ),
                    
                    html.H6("Sélectionnez les troubles:", className="mt-3"),
                    dcc.Checklist(
                        id='disorder-checklist',
                        options=[{'label': disorder_names[d], 'value': d} for d in disorders],
                        value=disorders,
                        inline=True
                    ),
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Section Vue d'ensemble mondiale
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Vue d'ensemble mondiale", className="card-header"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id='choropleth-map', style={'height': '600px'})
                        ], label="Carte mondiale", tab_id="tab-map"),
                        dbc.Tab([
                            dcc.Graph(id='radar-chart', style={'height': '600px'})
                        ], label="Profil par pays", tab_id="tab-radar"),
                        dbc.Tab([
                            dcc.Graph(id='heatmap-time', style={'height': '600px'})
                        ], label="Évolution géographique", tab_id="tab-heat")
                    ], id="overview-tabs", active_tab="tab-map")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Section Évolution temporelle
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Évolution temporelle", className="card-header"),
                dbc.CardBody([
                    dcc.Graph(id='time-series-chart', style={'height': '500px'})
                ]),
                dbc.CardFooter([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Statistiques de tendances", className="text-center")
                        ], width=12)
                    ]),
                    dbc.Row(id='sparklines-row')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Section Comparaisons entre troubles
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparaisons entre troubles", className="card-header"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id='stacked-bar-chart', style={'height': '500px'})
                        ], label="Barres empilées", tab_id="tab-stack"),
                        dbc.Tab([
                            dcc.Graph(id='treemap-chart', style={'height': '500px'})
                        ], label="Treemap", tab_id="tab-tree"),
                        dbc.Tab([
                            dcc.Graph(id='donut-chart', style={'height': '500px'})
                        ], label="Diagramme en anneau", tab_id="tab-donut"),
                        dbc.Tab([
                            dcc.Graph(id='bubble-chart', style={'height': '500px'})
                        ], label="Graphique à bulles", tab_id="tab-bubble")
                    ], id="comparison-tabs", active_tab="tab-stack")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Section Analyses spécifiques par trouble
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analyses spécifiques par trouble", className="card-header"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Sélectionnez un trouble:"),
                            dcc.Dropdown(
                                id='specific-disorder-dropdown',
                                options=[{'label': disorder_names[d], 'value': d} for d in disorders],
                                value=disorders[0],
                                clearable=False
                            )
                        ], width=12)
                    ], className="mb-3"),
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id='horizontal-bar-chart', style={'height': '500px'})
                        ], label="Top 15 pays", tab_id="tab-top15"),
                        dbc.Tab([
                            dcc.Graph(id='specific-heatmap', style={'height': '500px'})
                        ], label="Carte de chaleur", tab_id="tab-specific-heat"),
                        dbc.Tab([
                            dcc.Graph(id='boxplot-chart', style={'height': '500px'})
                        ], label="Distribution", tab_id="tab-box")
                    ], id="specific-tabs", active_tab="tab-top15")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Section Corrélations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Corrélations et Analyses Avancées", className="card-header"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id='correlation-matrix', style={'height': '500px'})
                        ], label="Matrice de corrélation", tab_id="tab-corr"),
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Sélectionnez deux troubles à comparer:"),
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='scatter-x-dropdown',
                                                options=[{'label': disorder_names[d], 'value': d} for d in disorders],
                                                value=disorders[0],
                                                clearable=False
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='scatter-y-dropdown',
                                                options=[{'label': disorder_names[d], 'value': d} for d in disorders],
                                                value=disorders[1],
                                                clearable=False
                                            )
                                        ], width=6)
                                    ]),
                                ], width=12)
                            ], className="mb-3"),
                            dcc.Graph(id='scatter-plot', style={'height': '500px'})
                        ], label="Graphique de dispersion", tab_id="tab-scatter")
                    ], id="correlation-tabs", active_tab="tab-corr")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Section Analyses régionales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analyses Régionales", className="card-header"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Sélectionnez l'année:"),
                            dcc.Slider(
                                id='regional-year-slider',
                                min=1970,
                                max=2019,
                                step=1,
                                marks={i: str(i) for i in range(1970, 2020, 10)},
                                value=2019
                            )
                        ], width=12)
                    ], className="mb-3"),
                    dcc.Graph(id='regional-bar-chart', style={'height': '500px'})
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Pied de page
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Dashboard créé avec Dash - Données sur les troubles de santé mentale de 1970 à 2019", 
                   className="text-center text-muted")
        ], width=12)
    ])
], fluid=True)

# Callbacks pour l'interactivité

# Afficher la plage d'années sélectionnée
@app.callback(
    Output('year-range-display', 'children'),
    Input('year-slider', 'value')
)
def update_year_display(years_selected):
    return f"Période sélectionnée: {years_selected[0]} - {years_selected[1]}"

# Carte choroplèthe mondiale
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('year-slider', 'value'),
     Input('specific-disorder-dropdown', 'value')]
)
def update_choropleth(years_selected, disorder):
    # Filtrer les données pour l'année la plus récente dans la plage sélectionnée
    year_data = df[df['Year'] == years_selected[1]]
    
    # Créer la carte choroplèthe
    fig = px.choropleth(
        year_data, 
        locations="Code",
        color=disorder,
        hover_name="Entity",
        projection="natural earth",
        color_continuous_scale=px.colors.sequential.Blues,
        title=f"Prévalence mondiale de {disorder_names[disorder]} en {years_selected[1]}"
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Prévalence (%)"
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    
    return fig

# Graphique radar pour les profils par pays
@app.callback(
    Output('radar-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_radar_chart(countries_selected, years_selected):
    if not countries_selected:
        return go.Figure()
    
    # Filtrer les données pour l'année la plus récente et les pays sélectionnés
    year_data = df[(df['Year'] == years_selected[1]) & (df['Entity'].isin(countries_selected))]
    
    # Créer le graphique radar
    fig = go.Figure()
    
    for country in countries_selected:
        country_data = year_data[year_data['Entity'] == country]
        if not country_data.empty:
            fig.add_trace(go.Scatterpolar(
                r=country_data[disorders].values.flatten().tolist(),
                theta=[disorder_names[d] for d in disorders],
                fill='toself',
                name=country
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, year_data[disorders].max().max() * 1.1]
            )
        ),
        title=f"Profil des troubles de santé mentale par pays en {years_selected[1]}",
        showlegend=True
    )
    
    return fig

# Carte de chaleur temporelle
@app.callback(
    Output('heatmap-time', 'figure'),
    [Input('specific-disorder-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('country-dropdown', 'value')]
)
def update_heatmap_time(disorder, years_selected, countries_selected):
    if not countries_selected or len(countries_selected) > 15:
        # Si trop de pays sont sélectionnés, prendre les 15 premiers
        countries_to_show = countries_selected[:15] if countries_selected else []
    else:
        countries_to_show = countries_selected
    
    # Filtrer les données pour la plage d'années et les pays sélectionnés
    filtered_data = df[(df['Year'] >= years_selected[0]) & 
                        (df['Year'] <= years_selected[1]) & 
                        (df['Entity'].isin(countries_to_show))]
    
    # Pivoter les données pour obtenir la forme appropriée pour la heatmap
    pivot_data = filtered_data.pivot(index='Entity', columns='Year', values=disorder)
    
    # Créer la carte de chaleur
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Année", y="Pays", color="Prévalence (%)"),
        title=f"Évolution de {disorder_names[disorder]} par pays de {years_selected[0]} à {years_selected[1]}",
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=years_selected[0],
            dtick=5
        )
    )
    
    return fig

# Graphique d'évolution temporelle
@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_time_series(years_selected, countries_selected, disorders_selected):
    if not countries_selected or not disorders_selected:
        return go.Figure()
    
    # Filtrer les données pour la plage d'années et les pays sélectionnés
    filtered_data = df[(df['Year'] >= years_selected[0]) & 
                        (df['Year'] <= years_selected[1]) & 
                        (df['Entity'].isin(countries_selected))]
    
    # Créer le graphique de série temporelle
    fig = go.Figure()
    
    for country in countries_selected:
        country_data = filtered_data[filtered_data['Entity'] == country]
        for disorder in disorders_selected:
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data[disorder],
                mode='lines+markers',
                name=f"{country} - {disorder_names[disorder]}",
                line=dict(color=colors[disorder]),
                marker=dict(size=6),
                legendgroup=country,
                showlegend=True
            ))
    
    fig.update_layout(
        title="Évolution des troubles de santé mentale au fil du temps",
        xaxis_title="Année",
        yaxis_title="Prévalence (%)",
        legend=dict(
            x=0,
            y=1,
            traceorder="grouped",
            font=dict(
                size=10
            )
        ),
        legend_title="Pays - Trouble",
        hovermode="x unified"
    )
    
    return fig

# Graphiques Sparklines pour les statistiques de tendances
@app.callback(
    Output('sparklines-row', 'children'),
    [Input('disorder-checklist', 'value'),
     Input('year-slider', 'value')]
)
def update_sparklines(disorders_selected, years_selected):
    if not disorders_selected:
        return []
    
    # Filtrer les données pour la plage d'années
    world_data = world_yearly_avg[(world_yearly_avg['Year'] >= years_selected[0]) & 
                                   (world_yearly_avg['Year'] <= years_selected[1])]
    
    sparkline_cols = []
    
    for disorder in disorders_selected:
        # Calculer les statistiques
        current_val = world_data[disorder].iloc[-1]
        change = current_val - world_data[disorder].iloc[0]
        pct_change = (change / world_data[disorder].iloc[0]) * 100 if world_data[disorder].iloc[0] > 0 else 0
        
        # Créer le mini graphique sparkline
        sparkfig = go.Figure()
        sparkfig.add_trace(go.Scatter(
            x=world_data['Year'],
            y=world_data[disorder],
            mode='lines',
            line=dict(color=colors[disorder], width=2),
            showlegend=False
        ))
        
        sparkfig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=50,
            width=120,
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Créer la colonne avec les statistiques et le sparkline
        col = dbc.Col([
            html.Div([
                html.H6(disorder_names[disorder], style={'fontSize': '0.9rem', 'marginBottom': '5px'}),
                html.Div([
                    html.Span(f"{current_val:.2f}%", style={'fontWeight': 'bold', 'fontSize': '1.1rem'}),
                    html.Span(
                        f" ({'+' if change >= 0 else ''}{change:.2f}%, {'+' if pct_change >= 0 else ''}{pct_change:.1f}%)",
                        style={'color': 'green' if change >= 0 else 'red', 'fontSize': '0.8rem'}
                    )
                ]),
                dcc.Graph(figure=sparkfig, config={'displayModeBar': False}, style={'height': '50px'})
            ], className="text-center p-2", style={'border': f'1px solid {colors[disorder]}', 'borderRadius': '5px'})
        ], width=12//len(disorders_selected) if len(disorders_selected) <= 6 else 4)
        
        sparkline_cols.append(col)
    
    # Organiser les colonnes en rangées si nécessaire
    if len(disorders_selected) <= 6:
        return sparkline_cols
    else:
        rows = []
        for i in range(0, len(sparkline_cols), 3):
            rows.append(dbc.Row(sparkline_cols[i:i+3], className="mb-3"))
        return rows

# Graphique à barres empilées
@app.callback(
    Output('stacked-bar-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_stacked_bar(years_selected, countries_selected, disorders_selected):
    if not countries_selected or not disorders_selected:
        return go.Figure()
    
    # Filtrer les données pour l'année la plus récente et les pays sélectionnés
    year_data = df[(df['Year'] == years_selected[1]) & (df['Entity'].isin(countries_selected))]
    
    # Créer le graphique à barres empilées
    fig = go.Figure()
    
    for disorder in disorders_selected:
        fig.add_trace(go.Bar(
            x=year_data['Entity'],
            y=year_data[disorder],
            name=disorder_names[disorder],
            marker_color=colors[disorder]
        ))
    
    fig.update_layout(
        title=f"Comparaison des troubles de santé mentale par pays en {years_selected[1]}",
        xaxis_title="Pays",
        yaxis_title="Prévalence (%)",
        barmode='stack',
        legend_title="Troubles",
        hovermode="closest"
    )
    
    return fig

# Treemap
@app.callback(
    Output('treemap-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_treemap(years_selected, countries_selected, disorders_selected):
    if not countries_selected or not disorders_selected:
        return go.Figure()
    
    # Filtrer les données pour l'année la plus récente et les pays sélectionnés
    year_data = df[(df['Year'] == years_selected[1]) & (df['Entity'].isin(countries_selected))]
    
    # Préparer les données pour le treemap
    treemap_data = []
    
    for country in countries_selected:
        country_data = year_data[year_data['Entity'] == country]
        if not country_data.empty:
            for disorder in disorders_selected:
                treemap_data.append({
                    'Country': country,
                    'Disorder': disorder_names[disorder],
                    'Value': country_data[disorder].values[0]
                })
    
    treemap_df = pd.DataFrame(treemap_data)
    
    # Créer le treemap
    fig = px.treemap(
        treemap_df,
        path=['Country', 'Disorder'],
        values='Value',
        color='Value',
        color_continuous_scale=px.colors.sequential.Blues,
        title=f"Répartition des troubles de santé mentale par pays en {years_selected[1]}"
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

# Diagramme en anneau
@app.callback(
    Output('donut-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_donut_chart(years_selected, countries_selected, disorders_selected):
    if not countries_selected or not disorders_selected or len(countries_selected) != 1:
        # Afficher un message si plus d'un pays est sélectionné
        fig = go.Figure()
        fig.add_annotation(
            text="Veuillez sélectionner un seul pays pour ce graphique",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Filtrer les données pour l'année la plus récente et le pays sélectionné
    country = countries_selected[0]
    year_data = df[(df['Year'] == years_selected[1]) & (df['Entity'] == country)]
    
    if year_data.empty:
        return go.Figure()
    
    # Calculer la somme totale pour normalisation (éviter plus de 100%)
    total_value = sum(year_data[disorder].values[0] for disorder in disorders_selected)
    
    # Créer le diagramme en anneau
    fig = go.Figure(data=[go.Pie(
        labels=[disorder_names[d] for d in disorders_selected],
        values=[year_data[d].values[0] for d in disorders_selected],
        hole=.4,
        marker_colors=[colors[d] for d in disorders_selected]
    )])
    
    fig.update_layout(
        title=f"Répartition des troubles de santé mentale pour {country} en {years_selected[1]}",
        annotations=[dict(text=f"{country}", x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

# Graphique à bulles
@app.callback(
    Output('bubble-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_bubble_chart(years_selected, countries_selected, disorders_selected):
    if not countries_selected or not disorders_selected or len(disorders_selected) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Veuillez sélectionner au moins deux troubles pour ce graphique",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Filtrer les données pour l'année la plus récente et les pays sélectionnés
    year_data = df[(df['Year'] == years_selected[1]) & (df['Entity'].isin(countries_selected))]
    
    # Sélectionner les deux premiers troubles pour les axes x et y
    x_disorder = disorders_selected[0]
    y_disorder = disorders_selected[1]
    
    # Utiliser le troisième trouble pour la taille des bulles, sinon utiliser la somme
    size_disorder = disorders_selected[2] if len(disorders_selected) > 2 else 'sum'
    
    # Calculer la taille des bulles
    if size_disorder == 'sum':
        year_data['bubble_size'] = year_data[disorders_selected].sum(axis=1)
        size_label = "Somme des troubles"
    else:
        year_data['bubble_size'] = year_data[size_disorder]
        size_label = disorder_names[size_disorder]
    
    # Créer le graphique à bulles
    fig = px.scatter(
        year_data,
        x=x_disorder,
        y=y_disorder,
        size='bubble_size',
        color='Entity',
        hover_name='Entity',
        size_max=50,
        title=f"Relation entre {disorder_names[x_disorder]} et {disorder_names[y_disorder]} en {years_selected[1]}"
    )
    
    fig.update_layout(
        xaxis_title=disorder_names[x_disorder],
        yaxis_title=disorder_names[y_disorder],
        legend_title="Pays",
        hovermode="closest"
    )
    
    return fig

# Graphique à barres horizontales pour le top 15 des pays
@app.callback(
    Output('horizontal-bar-chart', 'figure'),
    [Input('specific-disorder-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_horizontal_bar(disorder, years_selected):
    # Filtrer les données pour l'année la plus récente
    year_data = df[df['Year'] == years_selected[1]].copy()
    
    # Exclure les agrégats comme 'World', 'Europe', etc.
    individual_countries = year_data[~year_data['Entity'].isin(['World', 'Europe', 'Africa', 'Asia', 'North America', 'South America', 'Oceania'])]
    
    # Trier et prendre les 15 premiers pays
    top_countries = individual_countries.sort_values(by=disorder, ascending=False).head(15)
    
    # Créer le graphique à barres horizontales
    fig = px.bar(
        top_countries,
        y='Entity',
        x=disorder,
        orientation='h',
        color=disorder,
        color_continuous_scale=px.colors.sequential.Blues,
        title=f"Top 15 des pays avec la plus forte prévalence de {disorder_names[disorder]} en {years_selected[1]}"
    )
    
    fig.update_layout(
        yaxis=dict(title="Pays", autorange="reversed"),
        xaxis=dict(title="Prévalence (%)"),
        margin=dict(l=0, r=10, t=50, b=0)
    )
    
    return fig

# Carte de chaleur spécifique à un trouble
@app.callback(
    Output('specific-heatmap', 'figure'),
    [Input('specific-disorder-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('country-dropdown', 'value')]
)
def update_specific_heatmap(disorder, years_selected, countries_selected):
    if not countries_selected or len(countries_selected) > 20:
        # Si trop de pays sont sélectionnés, prendre les 20 premiers
        countries_to_show = countries_selected[:20] if countries_selected else []
    else:
        countries_to_show = countries_selected
    
    # Filtrer les données pour la plage d'années et les pays sélectionnés
    filtered_data = df[(df['Year'] >= years_selected[0]) & 
                        (df['Year'] <= years_selected[1]) & 
                        (df['Entity'].isin(countries_to_show))]
    
    # Créer une matrice pour la heatmap
    countries_array = []
    years_array = []
    values_array = []
    
    for country in countries_to_show:
        country_data = filtered_data[filtered_data['Entity'] == country]
        for year in range(years_selected[0], years_selected[1] + 1):
            year_country_data = country_data[country_data['Year'] == year]
            if not year_country_data.empty:
                countries_array.append(country)
                years_array.append(year)
                values_array.append(year_country_data[disorder].values[0])
    
    # Créer un DataFrame à partir des listes
    heatmap_df = pd.DataFrame({
        'Country': countries_array,
        'Year': years_array,
        'Value': values_array
    })
    
    # Pivoter le DataFrame pour la heatmap
    heatmap_pivot = heatmap_df.pivot(index='Country', columns='Year', values='Value')
    
    # Créer la carte de chaleur
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Blues',
        colorbar=dict(title="Prévalence (%)")
    ))
    
    fig.update_layout(
        title=f"Évolution de {disorder_names[disorder]} par pays et par année",
        xaxis=dict(title="Année", tickmode='linear', tick0=years_selected[0], dtick=5),
        yaxis=dict(title="Pays"),
        margin=dict(l=100, r=10, t=50, b=50)
    )
    
    return fig

# Diagramme en boîte pour la distribution des troubles
@app.callback(
    Output('boxplot-chart', 'figure'),
    [Input('specific-disorder-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('country-dropdown', 'value')]
)
def update_boxplot(disorder, years_selected, countries_selected):
    # Filtrer les données pour la plage d'années
    year_ranges = [
        (years_selected[0], years_selected[0] + (years_selected[1] - years_selected[0]) // 3),
        (years_selected[0] + (years_selected[1] - years_selected[0]) // 3 + 1, years_selected[0] + 2 * (years_selected[1] - years_selected[0]) // 3),
        (years_selected[0] + 2 * (years_selected[1] - years_selected[0]) // 3 + 1, years_selected[1])
    ]
    
    # Préparer les données pour le boxplot
    boxplot_data = []
    
    for start_year, end_year in year_ranges:
        period_label = f"{start_year}-{end_year}"
        filtered_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
        
        if countries_selected:
            filtered_data = filtered_data[filtered_data['Entity'].isin(countries_selected)]
        
        boxplot_data.append(
            go.Box(
                y=filtered_data[disorder],
                name=period_label,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(size=3)
            )
        )
    
    # Créer le diagramme en boîte
    fig = go.Figure(data=boxplot_data)
    
    fig.update_layout(
        title=f"Distribution de la prévalence de {disorder_names[disorder]} par période",
        yaxis=dict(title="Prévalence (%)"),
        xaxis=dict(title="Période"),
        boxmode='group',
        showlegend=False
    )
    
    return fig

# Matrice de corrélation
@app.callback(
    Output('correlation-matrix', 'figure'),
    [Input('year-slider', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_correlation_matrix(years_selected, disorders_selected):
    if not disorders_selected or len(disorders_selected) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Veuillez sélectionner au moins deux troubles pour ce graphique",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Filtrer les données pour la plage d'années
    filtered_data = df[(df['Year'] >= years_selected[0]) & (df['Year'] <= years_selected[1])]
    
    # Calculer la matrice de corrélation
    corr_matrix = filtered_data[disorders_selected].corr()
    
    # Créer les labels avec des noms plus courts
    labels = [disorder_names[d] for d in disorders_selected]
    
    # Créer la matrice de corrélation
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Coefficient de corrélation")
    ))
    
    # Ajouter les annotations avec les valeurs
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(round(value, 2)),
                    showarrow=False,
                    font=dict(color='white' if abs(value) > 0.5 else 'black')
                )
            )
    
    fig.update_layout(
        title="Matrice de corrélation entre les troubles de santé mentale",
        annotations=annotations
    )
    
    return fig

# Graphique de dispersion
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-dropdown', 'value'),
     Input('scatter-y-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('country-dropdown', 'value')]
)
def update_scatter_plot(x_disorder, y_disorder, years_selected, countries_selected):
    if x_disorder == y_disorder:
        fig = go.Figure()
        fig.add_annotation(
            text="Veuillez sélectionner deux troubles différents",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Filtrer les données pour l'année la plus récente
    year_data = df[df['Year'] == years_selected[1]].copy()
    
    if countries_selected:
        year_data = year_data[year_data['Entity'].isin(countries_selected)]
    
    # Calculer la ligne de régression
    slope, intercept = np.polyfit(year_data[x_disorder], year_data[y_disorder], 1)
    regression_y = slope * year_data[x_disorder] + intercept
    
    # Calculer le coefficient de corrélation
    corr_coef = np.corrcoef(year_data[x_disorder], year_data[y_disorder])[0, 1]
    
    # Créer le graphique de dispersion
    fig = px.scatter(
        year_data,
        x=x_disorder,
        y=y_disorder,
        hover_name='Entity',
        title=f"Relation entre {disorder_names[x_disorder]} et {disorder_names[y_disorder]} en {years_selected[1]}"
    )
    
    # Ajouter la ligne de régression
    fig.add_trace(go.Scatter(
        x=year_data[x_disorder],
        y=regression_y,
        mode='lines',
        name=f'Régression (r = {corr_coef:.2f})',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        xaxis=dict(title=disorder_names[x_disorder]),
        yaxis=dict(title=disorder_names[y_disorder]),
        legend=dict(x=0.02, y=0.98),
        hovermode="closest"
    )
    
    return fig

# Graphique à barres groupées pour l'analyse régionale
@app.callback(
    Output('regional-bar-chart', 'figure'),
    [Input('regional-year-slider', 'value'),
     Input('disorder-checklist', 'value')]
)
def update_regional_bar(year_selected, disorders_selected):
    if not disorders_selected:
        return go.Figure()
    
    # Définir les régions principales
    regions = ['World', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
    
    # Filtrer les données pour l'année sélectionnée et les régions
    regional_data = df[(df['Year'] == year_selected) & (df['Entity'].isin(regions))]
    
    # Créer le graphique à barres groupées
    fig = go.Figure()
    
    for disorder in disorders_selected:
        fig.add_trace(go.Bar(
            x=regional_data['Entity'],
            y=regional_data[disorder],
            name=disorder_names[disorder],
            marker_color=colors[disorder]
        ))
    
    fig.update_layout(
        title=f"Comparaison régionale des troubles de santé mentale en {year_selected}",
        xaxis=dict(title="Région"),
        yaxis=dict(title="Prévalence (%)"),
        barmode='group',
        legend_title="Troubles"
    )
    
    return fig

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)