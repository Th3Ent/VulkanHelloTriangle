-- premake5.lua
-- local project_location = path.getabsolute("/")

--Solution important settings
local workspace_name = "Gama"

--Projects global settings
local languague = "C++"
local projects_location = "./../build"

--Static lib project
local static_lib_project = "Gama_Engine" 

--Diferent projects
local num_of_projects = 1;
project_name_vulkan = {"Hello_Triangle"}

--Variables 
local i = 1

    workspace(workspace_name .. "Vulkan")
        startproject (project_name_vulkan[1])
        location (projects_location)
        platforms {"Win64"}
        configurations {"Debug" ,"Release"}

        includedirs {
            "./../include/",
            "./../deps/px/",
            "./../deps/glm/",
            "./../deps/tiny_obj/",
            "./../deps/stb/",
            "./../deps/imgui/",
            "./../deps/ImGuizmo/",
            "./../deps/Bullet3/Bullet3/",
            "./../deps/vulkan/include/",
            "./../deps/glfw/**", 
            "./../deps/lua/src/",
            "./../deps/minitrace/",
        }

        defines {
          "_CRT_SECURE_NO_WARNINGS", "_GLFW_WIN32", "WINDOWS_IMPL", "VULKAN_IMPL"
        }

        project (static_lib_project)

            location (projects_location .. "/"..static_lib_project)
            kind ("StaticLib")
            language (languague)
            cppdialect "C++17"
            targetdir "../bin/%{cfg.buildcfg}"
            debugdir  "../bin/%{cfg.buildcfg}"

            files { 

                "./../include/**.h", 


                "./../src/**.cpp", 

                "./../deps/vulkan/include/**.h",
                "./../deps/vulkan/include/**.c**",
                "./../deps/glfw/src/context.c",
                "./../deps/glfw/src/init.c ",
                "./../deps/glfw/src/input.c",
                "./../deps/glfw/src/monitor.c",
                "./../deps/glfw/src/vulkan.c",
                "./../deps/glfw/src/window.c",
                "./../deps/glfw/src/win32_init.c",
                "./../deps/glfw/src/win32_joystick.c",
                "./../deps/glfw/src/win32_monitor.c",
                "./../deps/glfw/src/win32_time.c",
                "./../deps/glfw/src/win32_thread.c",
                "./../deps/glfw/src/win32_window.c",
                "./../deps/glfw/src/wgl_context.c",
                "./../deps/glfw/src/egl_context.c",
                "./../deps/glfw/src/osmesa_context.c",
                "./../deps/glfw/**.h*",
                --"./../deps/Bullet3/Bullet3/**/**.h*",
                --"./../deps/Bullet3/Bullet3/**/**.c*",
                "./../deps/imgui/**.c*",
                "./../deps/imgui/**.h*",
                "./../deps/ImGuizmo/**.c*",
                "./../deps/ImGuizmo/**.h*",
                "./../include/EngineImgui/*.h",
                "./../include/OpenGL/*.h",
                "./../include/LuaBase.h",   
                "./../deps/minitrace/*.c",
                "./../deps/minitrace/*.h",

            }
            defines {"PX_SCHED_IMPLEMENTATION"}

            filter "configurations:Debug"
                defines {"DEBUG","MTR_ENABLED"}
                symbols "On"

            filter "configurations:Release"
                defines {"NDEBUG"}
                optimize "On"
           
    for _,i in ipairs(project_name_vulkan) do


        project (i)

            location (projects_location .. "/" .. i)
            kind ("ConsoleApp")
            language (languague)
            cppdialect "C++17"
            targetdir "./../bin/%{cfg.buildcfg}"
            debugdir  "./../bin/%{cfg.buildcfg}"

            files {("./../example/" .. i .. "/**.cpp") }

            libdirs{"./../deps/vulkan/Lib"}

            links {(static_lib_project)}
            links {"vulkan-1.lib"}

         filter "configurations:Debug"
             defines {"DEBUG","MTR_ENABLED"}
             symbols "On"
             flags {
               "NoPCH",
             }

         filter "configurations:Release"
             defines {"NDEBUG"}
             targetdir "../bin/"
             optimize "On"
             flags {
               "NoPCH",
             }
         
    end

