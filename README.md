# solr_py
<delete><query>*:*</query></delete>
curl 'http://localhost:8983/solr/techproducts/update?commit=true' --data-binary @example/exampledocs/books.json -H 'Content-type:application/json'

cat common_qa.txt | awk -F "\t" '{print $2}' > common_qa_q.txt