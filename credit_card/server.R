library(shiny)

server <- function(input, output) {
  output$plot1 <- renderPlot({ ggplot(data = data.frame(x = 1:10, y = rnorm(10)), aes(x = x, y = y)) + geom_line() })
  output$plot2 <- renderPlot({ ggplot(data = data.frame(x = 1:10, y = rnorm(10)), aes(x = x, y = y)) + geom_bar(stat = "identity") })
  output$plot3 <- renderPlot({ ggplot(data = data.frame(x = 1:10, y = rnorm(10)), aes(x = x, y = y)) + geom_histogram() })
  output$plot4 <- renderPlot({ ggplot(data = data.frame(x = 1:10, y = rnorm(10)), aes(x = x, y = y)) + geom_point() })
  
  # Batch scoring file processing
  output$table1 <- renderTable({
    inFile <- input$file1
    if (is.null(inFile))
      return()
    read.csv(inFile$datapath)
  })
  
  # Atomic scoring single record processing
  observeEvent(input$button, {
    data <- input$input1
    output$result <- renderText({
      paste("Resultado de predicción para:", data)
    })
  })
}


