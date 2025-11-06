




It often happens that I have several HTTP services and MCP services that I've developed I want to work with. I want to have them available ready to be used via python, chatGPT, and/or Claude. These can be AI agents or HTTP services. 
For now, we can assume that the AI agents are made with aw and aw_agents, which are our packages, so we can modify them to fit our needs. We can also assume that we made the HTTP services with FastAPI. 

The problem I'm trying to solve is that I don't want to have to constantly, manually, spin up a server when I need the services I need. I'd much rather have one or two (possibly one MCP server and one (fastAPI) HTTP server) running constantly, but with a means to do "CRUD" on the services. I'd like to be able to do things like add or remove a service, or possibly edit an existing one. 

What are the design patterns and best practices around this? 
Are there any third-party packages I should be using so as to not reinvent the wheel, or is it better for us to just implement this functionality ourselves?

