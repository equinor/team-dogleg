param (
    [switch]$build = $false,
    [switch]$b = $false
    
)
function build {
    docker build -t test . ; if ($?) {Write-Output "Success!"} else {Write-Output "Error!"}    
}
function run {
        
}
if ($build -or $b) {build}

else {    
    Write-Output("You must specify an action!")
}