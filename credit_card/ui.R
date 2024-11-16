
library(shiny)


ui <- dashboardPage(
  dashboardHeader(title = "Dashboard de Modelo"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Telemetría del Modelo", tabName = "telemetry", icon = icon("dashboard")),
      menuItem("Evaluación del Modelo", tabName = "evaluation", icon = icon("chart-line")),
      menuItem("Batch Scoring Test", tabName = "batch", icon = icon("table")),
      menuItem("Atomic Scoring Test", tabName = "atomic", icon = icon("database"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "telemetry",
              fluidRow(
                box(title = "Número de veces que el modelo es llamado", status = "primary", solidHeader = TRUE, plotOutput("plot1")),
                box(title = "Número de filas predichas", status = "warning", solidHeader = TRUE, plotOutput("plot2")),
                box(title = "Tiempo promedio de respuesta", status = "danger", solidHeader = TRUE, plotOutput("plot3"))
              )),
      tabItem(tabName = "evaluation",
              fluidRow(
                box(title = "Métricas y su historia", status = "info", solidHeader = TRUE, plotOutput("plot4"))
              )),
      tabItem(tabName = "batch",
              fluidRow(
                box(title = "Carga de un archivo para predecir", status = "primary", solidHeader = TRUE, fileInput("file1", "Choose CSV File", accept = ".csv")),
                box(title = "Output", status = "info", solidHeader = TRUE, tableOutput("table1"))
              )),
      tabItem(tabName = "atomic",
              fluidRow(
                box(title = "Predicción de un solo record", status = "success", solidHeader = TRUE, 
                    textInput("input1", "Ingrese los datos"),
                    actionButton("button", "Predecir"),
                    textOutput("result"))
              ))
    )
  )
)

