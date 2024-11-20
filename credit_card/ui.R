library(shiny)

fluidPage(
  titlePanel("Dashboard de Modelo"),
  
  tabsetPanel(type = "tabs",
              tabPanel("Telemetría del Modelo",
                       fluidRow(
                         column(12, renderPlotly("plotCalls")),
                         verbatimTextOutput("details")
                         )
                       
              ),
              tabPanel("Evaluación del Modelo",
                       fluidRow(
                         column(12, plotOutput("confusionMatrixPlot")),
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
                       sidebarLayout(
                         sidebarPanel(
                           h3("Ingrese los datos de la transacción"),
                           div(style = "overflow-y: scroll; height: 800px;",  # Ajusta el estilo como sea necesario
                               fluidRow(
                                 column(4, numericInput("time", "Time:", value = 0)),
                                 column(4, numericInput("v1", "V1:", value = 0)),
                                 column(4, numericInput("v2", "V2:", value = 0)),
                                 column(4, numericInput("v3", "V3:", value = 0)),
                                 column(4, numericInput("v4", "V4:", value = 0)),
                                 column(4, numericInput("v5", "V5:", value = 0)),
                                 column(4, numericInput("v6", "V6:", value = 0)),
                                 column(4, numericInput("v7", "V7:", value = 0)),
                                 column(4, numericInput("v8", "V8:", value = 0)),
                                 column(4, numericInput("v9", "V9:", value = 0)),
                                 column(4, numericInput("v10", "V10:", value = 0)),
                                 column(4, numericInput("v11", "V11:", value = 0)),
                                 column(4, numericInput("v12", "V12:", value = 0)),
                                 column(4, numericInput("v13", "V13:", value = 0)),
                                 column(4, numericInput("v14", "V14:", value = 0)),
                                 column(4, numericInput("v15", "V15:", value = 0)),
                                 column(4, numericInput("v16", "V16:", value = 0)),
                                 column(4, numericInput("v17", "V17:", value = 0)),
                                 column(4, numericInput("v18", "V18:", value = 0)),
                                 column(4, numericInput("v19", "V19:", value = 0)),
                                 column(4, numericInput("v20", "V20:", value = 0)),
                                 column(4, numericInput("v21", "V21:", value = 0)),
                                 column(4, numericInput("v22", "V22:", value = 0)),
                                 column(4, numericInput("v23", "V23:", value = 0)),
                                 column(4, numericInput("v24", "V24:", value = 0)),
                                 column(4, numericInput("v25", "V25:", value = 0)),
                                 column(4, numericInput("v26", "V26:", value = 0)),
                                 column(4, numericInput("v27", "V27:", value = 0)),
                                 column(4, numericInput("v28", "V28:", value = 0)),
                                 column(4, numericInput("amount", "Amount:", value = 0)),
                                 column(4, numericInput("class", "Class (0 = Normal, 1 = Fraud):", value = 0)),
                                 column(4, numericInput("hour", "Hour:", value = 0)),
                                 column(4, numericInput("fraud_spike", "Fraud Spike:", value = 0))
                               )
                           ),
                           actionButton("predict", "Predecir")
                         ),
                         mainPanel(
                           textOutput("result")
                         )
                       )
              )
  )
)

