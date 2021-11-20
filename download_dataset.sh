if [ ! -d "data" ]
then
    echo "Folder doesn't exist, creating now..."
    mkdir ./data
    echo "Folder created!"
else
    echo "Folder exists!"
fi

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            d) d=${VALUE} ;;
            *)
    esac
done

if [[ "$d" == "imdb" ]]
then
    cd ./data
    FILE="imdb.csv"

    if [ -f "$FILE" ]; then
        echo "$FILE exists!"
    else 
        echo "$FILE does not exist!"
        echo "Downloading IMDB dataset..."
        curl https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv -o imdb.csv
        echo "Complete downloading IMDB dataset!"
    fi
fi

if [[ "$d" == "yelp" ]]
then
    cd ./data
    FILE="yelp.csv"

    if [ -f "$FILE" ]; then
        echo "$FILE exists!"
    else 
        echo "$FILE does not exist!"
        echo "Downloading YELP dataset..."
        curl https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz -o yelp.csv.tgz
        tar -xvzf yelp.csv.tgz
        rm -r -f yelp.csv.tgz
        echo "Complete downloading YELP dataset!"
    fi
fi