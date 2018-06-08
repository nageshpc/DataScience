
import pdb
import json
import tweepy
import pickle
import sys


####
## some info
## user.id and user.screen_name -- there is a one to one mapping.. so can just use the latter 

##############################################################
def dump_user_info_to_csv (tweets, ofh):
    ## only the user info 
    for tweet in tweets :
        uinfo = tweet.user
        print( uinfo.id ,  uinfo.id_str  ,uinfo.name ,uinfo.screen_name , uinfo.location 
            , uinfo.protected ,uinfo.followers_count , uinfo.friends_count , uinfo.created_at
            , uinfo.favourites_count , uinfo.utc_offset , uinfo.time_zone , uinfo.geo_enabled 
            , uinfo.verified , uinfo.statuses_count , uinfo.lang , uinfo.contributors_enabled
            , uinfo.is_translator , uinfo.is_translation_enabled , uinfo.profile_background_color
            , uinfo.profile_background_image_url , uinfo.profile_background_image_url_https 
            , uinfo.profile_background_tile , uinfo.profile_image_url , uinfo.profile_image_url_https 
            , uinfo.profile_link_color , uinfo.profile_sidebar_border_color 
            , uinfo.profile_sidebar_fill_color , uinfo.profile_text_color , uinfo.profile_use_background_image 
            , uinfo.has_extended_profile , uinfo.default_profile , uinfo.default_profile_image , uinfo.following
            , uinfo.follow_request_sent , uinfo.notifications , uinfo.translator_type  
            , file= ofh, sep ="\t" )
            #, uinfo.profile_banner_url 

##############################################################
def dump_tweet_info_to_csv (  tweets, ofh ):
    ## only the user info 
    for tweet in tweets :
        try :
            pos_sen =  tweet.possibly_sensitive
            ##assert( pos_sen == "False" )
        except :
            pos_sen = "False"
            #print("missing possibly_sensitive", file=sys.stderr)
        ##
        print( 
                tweet.id, tweet.created_at 
                , tweet.in_reply_to_status_id 
                , tweet.in_reply_to_user_id
                , tweet.in_reply_to_screen_name 
                , tweet.geo, tweet.coordinates , tweet.place, tweet.contributors, tweet.is_quote_status, tweet.lang, pos_sen
                , tweet.retweet_count, tweet.favorite_count, tweet.favorited, tweet.retweeted
                , tweet.user.id
                , ",".join( [ht["text"] for ht in tweet.entities["hashtags"] ] ) # leaving this one out for now               , ",".join( tweet.entities["symbols"]) 
                , ",".join( [um["screen_name"] for um in tweet.entities["user_mentions"]] ) 
                , ",".join( [url["display_url"] for url in tweet.entities["urls"]] ) 
               , tweet.full_text.replace("\t", "    ").replace("\n", " ")
               , tweet.user.screen_name
             , file= ofh, sep ="\t")
           #, uinfo.profile_banner_url 

 
    
#############################################################
if __name__ == "__main__":
    with open('ml_tweets.bin',mode='rb') as file:
        tweets = pickle.load(file)
        ##
        #step 1... parse the tweets to get a CSV dump of twitter users
        #dump_user_info_to_csv(tweets , sys.stdout) 


        #step 2... get the tweet info -- leave out the user details 
        dump_tweet_info_to_csv( tweets, sys.stdout )

  
   
  
