library(shiny)

fluidPage(
  titlePanel("Dashboard de Modelo"),
  
  tabsetPanel(type = "tabs",
              tabPanel("Telemetría del Modelo",
                       fluidRow(
                         column(6, plotOutput("plot1")),
                         column(6, plotOutput("plot2")),
                         column(6, plotOutput("plot3"))
                       )
              ),
              tabPanel("Evaluación del Modelo",
                       fluidRow(
                         column(12, plotOutput("plot4")),
                         column(12, plotOutput("metrics_plot"))
                       )
              ),
              tabPanel("Registros API",
                       fluidRow(
                         column(12, dataTableOutput("api_logs_table"))
                       )
              ),
              tabPanel("Batch Scoring Test",
                       fluidRow(
                         column(6, fileInput("file1", "Choose CSV File", accept = ".csv")),
                         column(6, tableOutput("table1"))
                       )
              ),
              tabPanel("Atomic Scoring Test",
                       fluidRow(
                         column(6, textInput("input1", "Ingrese los datos")),
                         column(6, actionButton("button", "Predecir")),
                         column(12, textOutput("result"))
                       )
              )
  )
)

