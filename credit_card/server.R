library(shiny)
library(ggplot2)
library(DBI)
library(RPostgreSQL)


# Conectar 
con <- dbConnect(RPostgreSQL::PostgreSQL(),
                 dbname = "nombre_de_la_base_de_datos",
                 host = "host_de_la_base_de_datos",
                 port = 5432,
                 user = "usuario_de_la_base_de_datos",
                 password = "contraseña_del_usuario")


# Server 
server <- function(input, output, session) {
  output$plot1 <- renderPlot({
    data <- dbGetQuery(con, "x, y FROM datos_plot1")
    ggplot(data, aes(x = x, y = y)) + geom_line()
  })
  
  output$plot2 <- renderPlot({
    data <- dbGetQuery(con, "SELECT x, y FROM datos_plot2")
    ggplot(data, aes(x = x, y = y)) + geom_bar(stat = "identity")
  })
  
  output$plot3 <- renderPlot({
    data <- dbGetQuery(con, "SELECT x, y FROM tabla_datos_plot3")
    ggplot(data, aes(x = x, y = y)) + geom_histogram()
  })
  
  output$plot4 <- renderPlot({
    data <- dbGetQuery(con, "SELECT x, y FROM tabla_datos_plot4")
    ggplot(data, aes(x = x, y = y)) + geom_point()
  })
  
  
  # Tabla de registros API
  output$api_logs_table <- renderDataTable({
    dbGetQuery(con, "SELECT * FROM mage_ai.api_logs ORDER BY Timestamp DESC")
  })
  
  
  # Gráficos de métricas del modelo
  output$metrics_plot <- renderPlot({
    metrics_data <- dbGetQuery(con, "SELECT Timestamp, Accuracy, Precision, Recall, F1_Score FROM mage_ai.metrics ORDER BY Timestamp DESC")
    ggplot(metrics_data, aes(x = Timestamp)) + 
      geom_line(aes(y = Accuracy, color = "Accuracy")) +
      geom_line(aes(y = Precision, color = "Precision")) +
      geom_line(aes(y = Recall, color = "Recall")) +
      geom_line(aes(y = F1_Score, color = "F1_Score")) +
      labs(title = "Metrics Over Time", x = "Timestamp", y = "Score") +
      scale_color_manual(values = c("Accuracy" = "blue", "Precision" = "red", "Recall" = "green", "F1_Score" = "purple")) +
      theme_minimal()
  })
  
  output$table1 <- renderTable({
    inFile <- input$file1
    if (is.null(inFile)) {
      return(data.frame())
    }
    read.csv(inFile$datapath)
  })
  
  observeEvent(input$button, {
    data <- input$input1
    output$result <- renderText({
      paste("Resultado de predicción para:", data)
    })
  })  
  
  
  # Desconectar 
  session$onSessionEnded(function() {
    dbDisconnect(con)
  })
}


