param (
    # Mode switches
    [switch]$build = $false,
    [switch]$b = $false,
    [switch]$run = $false,
    [switch]$r = $false,

    # Arguments for python script
    $action = $args[0],
    $name = $args[1],
    $algorithm = $args[2],
    $timesteps = $args[3],
    $new_save_name = $args[4]        
)
$global:container_name = "auto_well_path"
$global:container_running_name = "auto_well_path_run"
$global:python_filename = "main.py"

function build_container {
    docker build -t $container_name . ; if ($?) {Write-Output "Success!"} else {Write-Output "Error!"}    
}

# A bad attempt at ignoring the return output form the delete command
function run_container {
    # We run with detached mode to avoid python image blocking
    try{
        delete_running($container_running_name)
    }
    catch{
        docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app --name $container_running_name $container_name
    }
    docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app --name $container_running_name $container_name   
}

function run_python_script($filename,$action,$name,$algorithm,$timesteps,$new_save_name){
    docker exec -it $container_running_name python $filename -u $action $name $algorithm $timesteps $new_save_name
}
function run {
    run_container ; if($?) {run_python_script $python_filename $action $name $algorithm $timesteps $new_save_name}    
}
function delete_running($name) {
     docker rm -f $name
}



if ($build -or $b) {build_container}
elseif ($run -or $r) {run}
else {    
    Write-Output("You must specify an action!")
}