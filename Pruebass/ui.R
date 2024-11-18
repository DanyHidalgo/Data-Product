#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)

library(shiny)

ui <- fluidPage(
  titlePanel("Dashboard de Modelo"),
  
  tabsetPanel(type = "tabs",
    tabPanel("Telemetría del Modelo",
      fluidRow(
        column(4, 
          box(title = "Número de veces que el modelo es llamado", status = "primary", solidHeader = TRUE, plotOutput("plot1")),
          box(title = "Número de filas predichas", status = "warning", solidHeader = TRUE, plotOutput("plot2")),
          box(title = "Tiempo promedio de respuesta", status = "danger", solidHeader = TRUE, plotOutput("plot3"))
        )
      )
    ),
    tabPanel("Evaluación del Modelo",
      fluidRow(
        column(4, 
          box(title = "Métricas y su historia", status = "info", solidHeader = TRUE, plotOutput("plot4"))
        )
      )
    ),
    tabPanel("Batch Scoring Test",
      fluidRow(
        column(4, 
          box(title = "Carga de un archivo para predecir", status = "primary", solidHeader = TRUE, fileInput("file1", "Choose CSV File", accept = ".csv")),
          box(title = "Output", status = "info", solidHeader = TRUE, tableOutput("table1"))
        )
      )
    ),
    tabPanel("Atomic Scoring Test",
      fluidRow(
        column(4, 
          box(title = "Predicción de un solo record", status = "success", solidHeader = TRUE, 
              textInput("input1", "Ingrese los datos"),
              actionButton("button", "Predecir"),
              textOutput("result"))
        )
      )
    )
  )
)

