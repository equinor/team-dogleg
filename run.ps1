# A powershell script to easier run Dockerized container
param (
    # Flag switches
    [switch]$build = $false,
    [switch]$b = $false,
    [switch]$run = $false,
    [switch]$r = $false,
    [switch]$auto = $false,
    [switch]$a = $false,
 
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
        docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app -p 0.0.0.0:6006:6006 -p  0.0.0.0:8988:8988 --name $container_running_name $container_name
    }
    docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app -p 0.0.0.0:6006:6006 -p  0.0.0.0:8988:8988 --name $container_running_name $container_name   
}

function run_python_script($filename,$action,$name,$algorithm,$timesteps,$new_save_name){
    docker exec -it $container_running_name python $filename $action $name $algorithm $timesteps $new_save_name
}
function run_tensorboard {
    docker exec -dit $container_running_name tensorboard --logdir /usr/src/app/tensorboard_logs --host 0.0.0.0 --port 6006 ; if ($?) {Write-Output "Tensorboard running at port 6006"}
}
function delete_running($name) {
     docker rm -f $name
}
function run {
    run_container ; if($?) {run_tensorboard} ; if ($?) {run_python_script $python_filename $action $name $algorithm $timesteps $new_save_name}    
}
function run2($script_action) {
    if ($script_action -eq "load") {
        Write-Output("Load")
    }
    elseif ($script_action -eq "train" -or $script_action -eq "retrain"){
        Write-Output("train")
    }
    else{
        Write-Output($script_action, "is not a valid argument!")
    }
}

if ($build -or $b) {build_container}
#elseif ($run -or $r) {run}
elseif ($run -or $r) {run2($action)}
elseif ($auto -or $a) {build_container ; if ($?) {run}}
else {    
    Write-Output("You must specify an action!")
}